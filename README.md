# OptiFund: AI-Powered ESG Portfolio Optimization (AI4ALL IGNITE Project, Summer 2025)

OptiFund is a reinforcement learning-based portfolio optimization system that combines traditional financial metrics with ESG (Environmental, Social, and Governance) considerations to create sustainable, high-performing investment portfolios.

## Features

- **AI-Powered Portfolio Optimization**: Uses Proximal Policy Optimization (PPO) to learn optimal asset allocation
- **ESG Integration**: Incorporates ESG risk scores into investment decisions
- **Multi-Asset Support**: Handles 491+ stocks with comprehensive market data
- **Risk Management**: Implements Sharpe ratio optimization, drawdown control, and diversification
- **Technical Analysis**: Uses moving averages, volatility, and momentum indicators
- **Transaction Cost Modeling**: Realistic trading costs and rebalancing penalties
- **Comprehensive Evaluation**: Multiple performance metrics and visualization

## Performance

After training and evaluation on 5 episodes:
- **Average Total Return:** 8.75% (short-term testing)
- **Average Sharpe Ratio:** 4.535 (excellent risk-adjusted returns)
- **Average Max Drawdown:** 4.85% (reasonable risk management)
- **Average Volatility:** 14.38% (low volatility)
- **Average ESG Score:** 0.557 (good ESG consideration) 

## Architecture

### Core Components

1. **Portfolio Environment** (`portfolio_env.py`)
   - Custom Gymnasium environment for portfolio optimization
   - 50,086 observation dimensions (market data + technical indicators + ESG)
   - 491 action dimensions (portfolio weights)
   - Multi-objective reward function

2. **Data Preprocessing** (`data_preprocessing.py`)
   - Loads and cleans stock price data
   - Merges with ESG risk ratings
   - Calculates technical indicators
   - Handles missing data and normalization

3. **RL Training** (`train_rl_agent.py`)
   - Full-featured training with PPO algorithm
   - Model checkpointing and evaluation
   - Performance visualization and analysis
   - Support for multiple RL algorithms

## File Structure

```
OptiFund/
├── portfolio_env.py          # Custom portfolio optimization environment
├── data_preprocessing.py     # Data loading and preprocessing
├── train_rl_agent.py        # Full RL training and evaluation
├── simple_train.py          # Quick training for testing
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── stock_details_5_years.csv    # Stock price data (5 years)
├── SP 500 ESG Risk Ratings.csv  # ESG risk scores
├── models/                  # Trained models and checkpoints
├── logs/                   # Training logs and TensorBoard files
└── optifund_env/           # Virtual environment
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd OptiFund
   ```

2. **Create a .env file**
   - Copy `.env.example` to `.env` and set the correct paths to your datasets:
     ```
     STOCK_DATA_PATH=stock_details_5_years.csv
     ESG_DATA_PATH=SP 500 ESG Risk Ratings.csv
     ```
   - (Do not commit your `.env` file; it is in .gitignore.)

3. **Create virtual environment**
   ```bash
   python3 -m venv optifund_env
   source optifund_env/bin/activate  # On Windows: optifund_env\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Add your datasets**
   - Place `stock_details_5_years.csv` in the root directory (or update the path in `.env`)
   - Place `SP 500 ESG Risk Ratings.csv` in the root directory (or update the path in `.env`)

## Usage

### 1. Data Preprocessing
To preprocess your data and check for errors, run:
```bash
python main.py
```
This will load, clean, and merge your stock and ESG datasets, and print a summary.

### 2. Environment Testing
To test the custom portfolio environment logic:
```bash
python test_env.py
```
This script runs a few random actions in the environment and prints out the results for debugging.

### 3. Quick RL Training
To quickly train and test a PPO agent:
```bash
python simple_train.py
```
This script runs a short training session and prints out basic performance metrics.

### 4. Full RL Training & Evaluation
For a full training run with evaluation and plots:
```bash
python train_rl_agent.py
```
This will train the agent, save checkpoints and the best model, and generate performance plots.

---

**Disclaimer:**  
This project is for research and educational purposes only. The code and results are not intended as financial advice, and should not be used to make real investment decisions. Always consult with a qualified financial professional before investing.