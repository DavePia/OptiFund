import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, returns_data, cov_matrix, esg_scores=None, initial_balance=10000, 
                 transaction_cost=0.001, risk_free_rate=0.02/252, window_size=20,
                 esg_weight=0.1, max_position=0.3, reward_scaling=1.0):
        """
        Initialize the portfolio optimization environment.
        
        Args:
            returns_data (pd.DataFrame): DataFrame of daily returns for each asset
            cov_matrix (pd.DataFrame): Covariance matrix of returns
            esg_scores (pd.Series, optional): ESG scores for each company. Defaults to None.
            initial_balance (float, optional): Initial portfolio value. Defaults to 10000.
            transaction_cost (float, optional): Cost per transaction. Defaults to 0.001.
            risk_free_rate (float, optional): Daily risk-free rate. Defaults to 0.02/252.
            window_size (int, optional): Lookback window for returns. Defaults to 20.
            esg_weight (float, optional): Weight for ESG component in reward. Defaults to 0.1.
            max_position (float, optional): Maximum position size per asset. Defaults to 0.3.
            reward_scaling (float, optional): Scaling factor for rewards. Defaults to 1.0.
        """
        super(PortfolioEnv, self).__init__()
        
        self.returns_data = returns_data
        self.cov_matrix = cov_matrix
        self.esg_scores = esg_scores if esg_scores is not None else pd.Series(0.5, index=returns_data.columns)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.esg_weight = esg_weight
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        
        self.n_assets = len(returns_data.columns)
        self.current_step = None
        self.portfolio_value = None
        self.weights = None
        self.history = None
        self.portfolio_values = []
        self.actions_taken = []
        
        # Calculate additional features
        self._calculate_technical_indicators()
        
        # Action space: portfolio weights (sum to 1, with max position constraint)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        
        # Enhanced observation space: past returns + technical indicators + current weights + ESG scores + market features
        obs_size = (self.n_assets * self.window_size * 5 +  # 5 technical indicators (returns, ma_short, ma_long, volatility, momentum)
                   self.n_assets +  # current weights
                   1 +  # normalized portfolio value
                   self.n_assets +  # ESG scores
                   3)  # market features (volatility, correlation, momentum)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.reset()
    
    def _calculate_technical_indicators(self):
        """Calculate technical indicators for each asset"""
        # Calculate moving averages
        self.ma_short = self.returns_data.rolling(window=5).mean()
        self.ma_long = self.returns_data.rolling(window=20).mean()
        
        # Calculate volatility (rolling standard deviation)
        self.volatility = self.returns_data.rolling(window=20).std()
        
        # Calculate momentum (cumulative returns over window)
        self.momentum = self.returns_data.rolling(window=10).sum()
        
        # Fill NaN values
        self.ma_short = self.ma_short.fillna(0)
        self.ma_long = self.ma_long.fillna(0)
        self.volatility = self.volatility.fillna(0)
        self.momentum = self.momentum.fillna(0)
    
    def _get_market_features(self):
        """Get market-wide features"""
        if self.current_step < self.window_size:
            return np.array([0.0, 0.0, 0.0])
        
        # Market volatility (average of asset volatilities)
        current_vol = self.volatility.iloc[self.current_step].mean()
        
        # Market correlation (average pairwise correlation)
        returns_window = self.returns_data.iloc[self.current_step - self.window_size:self.current_step]
        corr_matrix = returns_window.corr()
        market_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        # Market momentum (average momentum across assets)
        market_momentum = self.momentum.iloc[self.current_step].mean()
        
        return np.array([current_vol, market_corr, market_momentum])
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.weights = np.ones(self.n_assets) / self.n_assets  # Equal initial weights
        self.history = []
        self.portfolio_values = [self.initial_balance]
        self.actions_taken = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step < self.window_size:
            # Pad with zeros if not enough history
            padding_size = self.window_size - self.current_step
            past_returns = np.zeros(self.n_assets * self.window_size)
            past_ma_short = np.zeros(self.n_assets * self.window_size)
            past_ma_long = np.zeros(self.n_assets * self.window_size)
            past_volatility = np.zeros(self.n_assets * self.window_size)
            past_momentum = np.zeros(self.n_assets * self.window_size)
        else:
            # Get past returns for each asset
            past_returns = self.returns_data.iloc[
                self.current_step - self.window_size:self.current_step
            ].values.flatten()
            
            # Get technical indicators
            past_ma_short = self.ma_short.iloc[
                self.current_step - self.window_size:self.current_step
            ].values.flatten()
            
            past_ma_long = self.ma_long.iloc[
                self.current_step - self.window_size:self.current_step
            ].values.flatten()
            
            past_volatility = self.volatility.iloc[
                self.current_step - self.window_size:self.current_step
            ].values.flatten()
            
            past_momentum = self.momentum.iloc[
                self.current_step - self.window_size:self.current_step
            ].values.flatten()
        
        # Get ESG scores for each asset
        esg_scores = self.esg_scores.values
        
        # Get market features
        market_features = self._get_market_features()
        
        # Combine all features
        obs = np.concatenate([
            past_returns,
            past_ma_short,
            past_ma_long,
            past_volatility,
            past_momentum,
            self.weights,
            [self.portfolio_value / self.initial_balance],  # Normalized portfolio value
            esg_scores,
            market_features
        ])
        
        return obs.astype(np.float32)
    
    def _normalize_weights(self, action):
        """Normalize weights using softmax and apply position limits"""
        # Apply softmax to get valid probabilities
        exp_action = np.exp(action - np.max(action))
        weights = exp_action / (exp_action.sum() + 1e-10)
        
        # Apply position limits
        weights = np.clip(weights, 0, self.max_position)
        
        # Renormalize to sum to 1
        weights = weights / (weights.sum() + 1e-10)
        
        return weights
    
    def step(self, action):
        """Execute one step in the environment"""
        # Normalize action to valid portfolio weights
        new_weights = self._normalize_weights(action)
        
        # Calculate transaction costs
        weight_diff = np.abs(new_weights - self.weights).sum()
        transaction_cost = self.transaction_cost * weight_diff * self.portfolio_value
        
        # Update portfolio value after transaction costs
        self.portfolio_value -= transaction_cost
        
        # Get returns for current step
        current_returns = self.returns_data.iloc[self.current_step].values
        
        # Update portfolio value based on returns
        new_portfolio_value = self.portfolio_value * (1 + np.dot(new_weights, current_returns))
        
        # Calculate financial metrics
        portfolio_return = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        portfolio_volatility = np.sqrt(np.dot(new_weights.T, np.dot(self.cov_matrix, new_weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / (portfolio_volatility + 1e-10)
        
        # Calculate ESG metrics
        esg_score = np.dot(new_weights, self.esg_scores.values)
        
        # Enhanced reward function
        # Base reward: Sharpe ratio
        base_reward = sharpe_ratio
        
        # ESG contribution
        esg_reward = self.esg_weight * esg_score
        
        # Transaction cost penalty
        transaction_penalty = -0.1 * (weight_diff / 2)  # Penalize excessive trading
        
        # Diversification bonus (penalize concentration)
        concentration_penalty = -0.05 * (np.max(new_weights) - 1/self.n_assets)
        
        # Combined reward
        reward = (base_reward + esg_reward + transaction_penalty + concentration_penalty) * self.reward_scaling
        
        # Update state
        self.weights = new_weights
        self.portfolio_value = new_portfolio_value
        self.current_step += 1
        
        # Store history
        self.portfolio_values.append(self.portfolio_value)
        self.actions_taken.append(new_weights.copy())
        
        # Check if done
        terminated = self.current_step >= len(self.returns_data) - 1
        truncated = False  # No time/truncation limit for now
        
        # Calculate additional metrics
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        max_drawdown = self._calculate_max_drawdown()
        
        # Store info
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights,
            'sharpe_ratio': sharpe_ratio,
            'esg_score': esg_score,
            'transaction_cost': transaction_cost,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'portfolio_volatility': portfolio_volatility,
            'step': self.current_step
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from portfolio values"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        peak = self.portfolio_values[0]
        max_dd = 0.0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def get_portfolio_stats(self):
        """Get comprehensive portfolio statistics"""
        if len(self.portfolio_values) < 2:
            return {}
        
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        stats = {
            'total_return': (self.portfolio_values[-1] - self.initial_balance) / self.initial_balance,
            'annualized_return': np.mean(returns) * 252,
            'volatility': np.std(returns) * np.sqrt(252),
            'sharpe_ratio': (np.mean(returns) * 252 - self.risk_free_rate * 252) / (np.std(returns) * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(),
            'final_value': self.portfolio_values[-1],
            'num_trades': len([w for w in self.actions_taken if np.any(w != self.actions_taken[0])])
        }
        
        return stats
    
    def render(self, mode='human'):
        """Render the current state"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Value: ${self.portfolio_value:,.2f}")
            print(f"Total Return: {((self.portfolio_value - self.initial_balance) / self.initial_balance) * 100:.2f}%")
            print(f"Current Weights: {np.round(self.weights, 3)}")
            print(f"Current ESG Score: {np.dot(self.weights, self.esg_scores.values):.3f}")
            print(f"Max Position: {np.max(self.weights):.3f}")
            print("-" * 50)