import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from portfolio_env import PortfolioEnv

class PortfolioTrainer:
    def __init__(self, data_dir='.', model_dir='models', log_dir='logs'):
        """
        Initialize the portfolio trainer.
        
        Args:
            data_dir (str): Directory containing data files
            model_dir (str): Directory to save trained models
            log_dir (str): Directory to save training logs
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize data
        self.preprocessor = None
        self.env = None
        self.model = None
        
    def load_data(self, split_ratio=0.8):
        """Load and preprocess data, then split into train/test sets by date."""
        print("Loading and preprocessing data...")
        self.preprocessor = DataPreprocessor()
        processed_data = self.preprocessor.preprocess_all()
        returns = processed_data['returns']
        esg_scores = processed_data['esg_scores']
        # Split by date
        split_idx = int(len(returns) * split_ratio)
        self.train_returns = returns.iloc[:split_idx]
        self.test_returns = returns.iloc[split_idx:]
        self.train_cov = self.train_returns.cov()
        self.test_cov = self.test_returns.cov()
        self.train_esg = esg_scores.loc[self.train_returns.columns]
        self.test_esg = esg_scores.loc[self.test_returns.columns]
        print(f"Train set: {len(self.train_returns)} steps, Test set: {len(self.test_returns)} steps")
        print(f"Train date range: {self.train_returns.index[0]} to {self.train_returns.index[-1]}")
        print(f"Test date range: {self.test_returns.index[0]} to {self.test_returns.index[-1]}")

    def create_environment(self, env_params=None, mode='train'):
        """Create the portfolio environment for train or test set."""
        if env_params is None:
            env_params = {
                'initial_balance': 10000,
                'transaction_cost': 0.002,  # More realistic
                'risk_free_rate': 0.02/252,
                'window_size': 20,
                'esg_weight': 0.1,
                'max_position': 0.3,
                'reward_scaling': 1.0
            }
        print(f"Creating portfolio environment for {mode} set...")
        if mode == 'train':
            returns_data = self.train_returns
            cov_matrix = self.train_cov
            esg_scores = self.train_esg
        else:
            returns_data = self.test_returns
            cov_matrix = self.test_cov
            esg_scores = self.test_esg
        self.env = PortfolioEnv(
            returns_data=returns_data,
            cov_matrix=cov_matrix,
            esg_scores=esg_scores,
            **env_params
        )
        self.env = Monitor(self.env)
        print(f"Environment created with {self.env.observation_space.shape[0]} observation dimensions")
        print(f"Action space: {self.env.action_space.shape[0]} dimensions")
        
    def train_agent(self, algorithm='PPO', total_timesteps=100000, **kwargs):
        """
        Train a reinforcement learning agent.
        
        Args:
            algorithm (str): RL algorithm to use ('PPO', 'A2C', 'SAC')
            total_timesteps (int): Total training timesteps
            **kwargs: Additional parameters for the algorithm
        """
        if self.env is None:
            raise ValueError("Environment not created. Call create_environment() first.")
        
        print(f"Training {algorithm} agent for {total_timesteps} timesteps...")
        
        # Algorithm-specific parameters
        if algorithm == 'PPO':
            model_params = {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'policy_kwargs': {
                    'net_arch': [dict(pi=[256, 256], vf=[256, 256])]
                }
            }
            model_class = PPO
        elif algorithm == 'A2C':
            model_params = {
                'learning_rate': 7e-4,
                'n_steps': 5,
                'gamma': 0.99,
                'gae_lambda': 1.0,
                'ent_coef': 0.01,
                'vf_coef': 0.25,
                'max_grad_norm': 0.5,
                'policy_kwargs': {
                    'net_arch': [dict(pi=[256, 256], vf=[256, 256])]
                }
            }
            model_class = A2C
        elif algorithm == 'SAC':
            model_params = {
                'learning_rate': 3e-4,
                'buffer_size': 1000000,
                'learning_starts': 100,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'ent_coef': 'auto',
                'policy_kwargs': {
                    'net_arch': [256, 256]
                }
            }
            model_class = SAC
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Update with user-provided parameters
        model_params.update(kwargs)
        
        # Create model
        self.model = model_class(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=self.log_dir,
            **model_params
        )
        
        # Setup callbacks
        eval_env = PortfolioEnv(
            returns_data=self.returns_data,
            cov_matrix=self.cov_matrix,
            esg_scores=self.esg_scores,
            initial_balance=10000,
            transaction_cost=0.001,
            risk_free_rate=0.02/252,
            window_size=20,
            esg_weight=0.1,
            max_position=0.3,
            reward_scaling=1.0
        )
        eval_env = Monitor(eval_env)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.model_dir}/best_{algorithm.lower()}",
            log_path=f"{self.log_dir}/{algorithm.lower()}_eval",
            eval_freq=max(total_timesteps // 10, 1),
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(total_timesteps // 5, 1),
            save_path=f"{self.model_dir}/{algorithm.lower()}_checkpoints",
            name_prefix=f"{algorithm.lower()}_model"
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save final model
        final_model_path = f"{self.model_dir}/{algorithm.lower()}_final"
        self.model.save(final_model_path)
        print(f"Training completed! Model saved to {final_model_path}")
        
    def evaluate_agent(self, model_path=None, num_episodes=10):
        """
        Evaluate the trained agent.
        
        Args:
            model_path (str): Path to the model to evaluate
            num_episodes (int): Number of episodes to run
        """
        if model_path is None and self.model is None:
            raise ValueError("No model available for evaluation.")
        
        if model_path:
            # Load model
            if 'PPO' in model_path:
                model = PPO.load(model_path)
            elif 'A2C' in model_path:
                model = A2C.load(model_path)
            elif 'SAC' in model_path:
                model = SAC.load(model_path)
            else:
                raise ValueError("Could not determine model type from path.")
        else:
            model = self.model
        
        print(f"Evaluating agent over {num_episodes} episodes...")
        
        results = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_info = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                episode_reward += reward
                episode_info.append(info)
            
            # Get final portfolio stats - handle Monitor wrapper
            if hasattr(self.env, 'env'):
                # Environment is wrapped (e.g., Monitor)
                final_stats = self.env.env.get_portfolio_stats()
            else:
                # Direct environment
                final_stats = self.env.get_portfolio_stats()
            # Added ESG score to results (final portfolio ESG score)
            final_stats['esg_score'] = info.get('esg_score', float('nan')) if isinstance(info, dict) else float('nan')
            final_stats['episode'] = episode
            final_stats['total_reward'] = episode_reward
            results.append(final_stats)
            
            print(f"Episode {episode + 1}: Return = {final_stats['total_return']:.2%}, "
                  f"Sharpe = {final_stats['sharpe_ratio']:.3f}, "
                  f"Max DD = {final_stats['max_drawdown']:.2%}")
        
        # Aggregate results
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Average Total Return: {results_df['total_return'].mean():.2%} ± {results_df['total_return'].std():.2%}")
        print(f"Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.3f} ± {results_df['sharpe_ratio'].std():.3f}")
        print(f"Average Max Drawdown: {results_df['max_drawdown'].mean():.2%} ± {results_df['max_drawdown'].std():.2%}")
        print(f"Average Volatility: {results_df['volatility'].mean():.2%} ± {results_df['volatility'].std():.2%}")
        print(f"Average ESG Score: {results_df['esg_score'].mean():.3f} ± {results_df['esg_score'].std():.3f}")
        print(f"Average Final Value: ${results_df['final_value'].mean():,.0f}")
        
        return results_df
    
    def plot_results(self, results_df, save_path=None):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total returns distribution
        axes[0, 0].hist(results_df['total_return'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(results_df['total_return'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 0].set_title('Distribution of Total Returns')
        axes[0, 0].set_xlabel('Total Return')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Sharpe ratio distribution
        axes[0, 1].hist(results_df['sharpe_ratio'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(results_df['sharpe_ratio'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 1].set_title('Distribution of Sharpe Ratios')
        axes[0, 1].set_xlabel('Sharpe Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Max drawdown distribution
        axes[1, 0].hist(results_df['max_drawdown'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 0].axvline(results_df['max_drawdown'].mean(), color='red', linestyle='--', label='Mean')
        axes[1, 0].set_title('Distribution of Maximum Drawdowns')
        axes[1, 0].set_xlabel('Max Drawdown')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Return vs Risk scatter plot
        axes[1, 1].scatter(results_df['volatility'], results_df['total_return'], alpha=0.6, color='purple')
        axes[1, 1].set_title('Risk vs Return')
        axes[1, 1].set_xlabel('Volatility')
        axes[1, 1].set_ylabel('Total Return')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to {save_path}")
        
        plt.show()

def main():
    trainer = PortfolioTrainer()
    trainer.load_data()
    # Train on train set
    trainer.create_environment(mode='train')
    print("\n" + "="*50)
    print("TRAINING PPO AGENT")
    print("="*50)
    trainer.train_agent(
        algorithm='PPO',
        total_timesteps=100_000,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128
    )
    # Evaluate on test set
    print("\n" + "="*50)
    print("EVALUATING TRAINED AGENT ON TEST SET")
    print("="*50)
    trainer.create_environment(mode='test')
    results = trainer.evaluate_agent(num_episodes=10)
    trainer.plot_results(results, save_path='training_results.png')
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main() 