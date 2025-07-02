from data_preprocessing import DataPreprocessor
from portfolio_env import PortfolioEnv
import numpy as np

def main():
    # Initialize the data preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess all data
    processed_data = preprocessor.preprocess_all()
    
    # Extract the processed data
    returns_data = processed_data['returns']
    cov_matrix = processed_data['cov_matrix']
    esg_scores = processed_data['esg_scores']
    
    # Create the portfolio environment
    env = PortfolioEnv(
        returns_data=returns_data,
        cov_matrix=cov_matrix,
        esg_scores=esg_scores,
        initial_balance=10000,
        transaction_cost=0.001,
        esg_weight=0.1
    )
    
    # Example usage - run a few steps
    obs = env.reset()
    print("Environment initialized successfully!")
    print(f"Number of assets: {env.n_assets}")
    print(f"Initial portfolio value: ${env.portfolio_value:,.2f}")
    
    # Run a few steps as an example
    for step in range(5):
        # Random action (in practice, you'd use your RL agent here)
        action = np.random.dirichlet(np.ones(env.n_assets))
        obs, reward, done, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"Portfolio Value: ${info['portfolio_value']:,.2f}")
        print(f"Sharpe Ratio: {info['sharpe_ratio']:.4f}")
        print(f"ESG Score: {info['esg_score']:.4f}")
        print(f"Reward: {reward:.4f}")
        
        if done:
            break

if __name__ == "__main__":
    main() 