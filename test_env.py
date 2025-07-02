from portfolio_env import PortfolioEnv
from data_preprocessing import DataPreprocessor
import numpy as np

def test_environment():
    print("Testing enhanced portfolio environment...")
    
    # Load data
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_all()
    
    # Create environment
    env = PortfolioEnv(
        data['returns'], 
        data['cov_matrix'], 
        data['esg_scores']
    )
    
    print(f"Environment created successfully!")
    print(f"Number of assets: {env.n_assets}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    # Test a few steps
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for step in range(3):
        # Random action
        action = np.random.randn(env.n_assets)
        obs, reward, done, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio value: ${info['portfolio_value']:,.2f}")
        print(f"  Sharpe ratio: {info['sharpe_ratio']:.4f}")
        print(f"  ESG score: {info['esg_score']:.4f}")
        print(f"  Total return: {info['total_return']:.2%}")
        
        if done:
            break
    
    # Get final stats
    stats = env.get_portfolio_stats()
    print(f"\nFinal portfolio stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    test_environment() 