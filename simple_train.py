from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np
import pandas as pd

from data_preprocessing import DataPreprocessor
from portfolio_env import PortfolioEnv

def simple_training():
    print("Starting simple RL training...")
    
    # Load data
    print("Loading data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_all()
    
    # Create environment
    print("Creating environment...")
    env = PortfolioEnv(
        data['returns'], 
        data['cov_matrix'], 
        data['esg_scores'],
        initial_balance=10000,
        transaction_cost=0.001,
        window_size=20,
        esg_weight=0.1,
        max_position=0.3
    )
    
    # Wrap in Monitor for logging
    env = Monitor(env)
    
    print(f"Environment created with {env.observation_space.shape[0]} observation dimensions")
    print(f"Action space: {env.action_space.shape[0]} dimensions")
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )
    
    # Train for a small number of timesteps
    print("Training model...")
    model.learn(total_timesteps=10000, progress_bar=True)
    
    # Save the model
    model.save("ppo_portfolio_model")
    print("Model saved as 'ppo_portfolio_model'")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_env = PortfolioEnv(
        data['returns'], 
        data['cov_matrix'], 
        data['esg_scores'],
        initial_balance=10000,
        transaction_cost=0.001,
        window_size=20,
        esg_weight=0.1,
        max_position=0.3
    )
    
    obs, _ = test_env.reset()  # Get observation and ignore info
    done = False
    total_reward = 0
    step_count = 0
    
    while not done and step_count < 100:  # Limit to 100 steps for testing
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        if step_count % 20 == 0:
            print(f"Step {step_count}: Portfolio Value = ${info['portfolio_value']:,.2f}, "
                  f"Sharpe = {info['sharpe_ratio']:.3f}, ESG = {info['esg_score']:.3f}")
    
    # Get final stats
    final_stats = test_env.get_portfolio_stats()
    print(f"\nFinal Results:")
    print(f"Total Return: {final_stats['total_return']:.2%}")
    print(f"Sharpe Ratio: {final_stats['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {final_stats['max_drawdown']:.2%}")
    print(f"Final Portfolio Value: ${final_stats['final_value']:,.2f}")
    print(f"Total Reward: {total_reward:.3f}")
    
    print("\nSimple training completed!")

if __name__ == "__main__":
    simple_training() 