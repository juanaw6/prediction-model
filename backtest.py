import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

from test_ft_model import EnhancedFinancialTransformer

# You'll need to import your model class from your module
# from test_ft_model import EnhancedFinancialTransformer

# Define load functions that were referenced but not defined
def load_trained_model():
    """Load the trained model"""
    feature_dim = 5  # Number of features (OHLCV)
    model = EnhancedFinancialTransformer(
        feature_dim=feature_dim,
        dropout=0.2,
        weight_decay=1e-5
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('best_financial_transformer.pth', map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

def load_historical_data():
    """Load and prepare historical data"""
    # Replace this with your actual data loading logic
    csv_data_path = './data/raw/solusdt_5m_2024_2025.csv'
    df = pd.read_csv(csv_data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def prepare_data_for_prediction(new_df, scaler=None):
    """
    Prepare new data for prediction.
    
    Args:
        new_df: DataFrame with at least 'open', 'high', 'low', 'close', 'volume' columns
        scaler: Optional pre-fitted StandardScaler. If None, a new one will be created
    
    Returns:
        Processed features ready for model input and the scaler used
    """
    features = new_df[['open', 'high', 'low', 'close', 'volume']].values
    
    if scaler is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = scaler.transform(features)
    
    scaled_features = np.nan_to_num(scaled_features)
    return scaled_features, scaler

def predict_next_return(model, features, seq_length=60):
    """
    Make a prediction for the next period's return
    
    Args:
        model: Trained transformer model
        features: Preprocessed feature array
        seq_length: Sequence length the model was trained on
    
    Returns:
        Predicted return for the next period
    """
    # Make sure we have enough data points
    if len(features) < seq_length:
        raise ValueError(f"Need at least {seq_length} data points, but got {len(features)}")
    
    # Get the last sequence
    latest_sequence = features[-seq_length:]
    
    # Convert to tensor
    X = torch.tensor(latest_sequence, dtype=torch.float32).unsqueeze(0)
    X = X.to(next(model.parameters()).device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(X)
    
    return prediction.item()

def backtest(model, test_data, scaler=None, initial_capital=10000, threshold=0.001):
    """
    Backtest the model on historical data
    
    Args:
        model: Trained model
        test_data: DataFrame with test data
        scaler: Optional pre-fitted scaler
        initial_capital: Starting capital
        threshold: Threshold for buy/sell signals
    
    Returns:
        DataFrame with backtest results
    """
    # Make a copy to avoid modifying original data
    test_data = test_data.copy()
    
    # Reset index if it's a DatetimeIndex to have a regular column for date
    if isinstance(test_data.index, pd.DatetimeIndex):
        test_data = test_data.reset_index()
    
    features, scaler = prepare_data_for_prediction(test_data, scaler)
    
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long position
    entry_price = 0
    
    results = []
    seq_length = 60
    trades = 0
    profitable_trades = 0
    
    for i in range(seq_length, len(features)-1):
        # Get sequence and make prediction
        sequence = features[i-seq_length:i]
        X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        X = X.to(next(model.parameters()).device)
        
        with torch.no_grad():
            predicted_return = model(X).item()
        
        # Current price and actual next return
        current_price = test_data['close'].iloc[i]
        next_price = test_data['close'].iloc[i+1]
        actual_return = (next_price - current_price) / current_price
        
        # Trading logic
        signal = None
        pnl = 0
        
        if predicted_return > threshold and position == 0:
            # Buy
            position = 1
            signal = "BUY"
            entry_price = current_price
        elif predicted_return < -threshold and position == 1:
            # Sell
            pnl = ((next_price / entry_price) - 1) * 100  # PnL as percentage
            trades += 1
            if next_price > entry_price:
                profitable_trades += 1
            
            position = 0
            signal = "SELL"
            capital *= (1 + actual_return)  # Update capital based on actual return
        elif position == 1:
            # Holding position
            capital *= (1 + actual_return)  # Update capital while holding
            
        # Determine date field based on whether we have a timestamp column or index
        date_value = test_data['timestamp'].iloc[i] if 'timestamp' in test_data.columns else i
        
        results.append({
            'date': date_value,
            'price': current_price,
            'predicted_return': predicted_return,
            'actual_return': actual_return,
            'signal': signal,
            'position': position,
            'capital': capital,
            'pnl': pnl
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate win rate
    win_rate = (profitable_trades / trades * 100) if trades > 0 else 0
    
    print(f"Total trades: {trades}")
    print(f"Profitable trades: {profitable_trades}")
    print(f"Win rate: {win_rate:.2f}%")
    
    return results_df

def trading_system(model, historical_data, scaler):
    """Simple implementation of trading system for demonstration"""
    print("Trading system simulation started")
    print("This would normally connect to an exchange API")
    print("For now, just running a sample prediction on the latest data")
    
    features, _ = prepare_data_for_prediction(historical_data, scaler)
    pred = predict_next_return(model, features)
    
    print(f"Prediction for next period return: {pred:.6f}")
    
    if pred > 0.001:
        print("SIGNAL: BUY")
    elif pred < -0.001:
        print("SIGNAL: SELL")
    else:
        print("SIGNAL: HOLD")
    
    print("In a real system, this would execute an order via API")

def monitor_model_performance(predictions, actuals, threshold=0.05):
    """
    Monitor model performance and determine if retraining is needed
    
    Args:
        predictions: Array of recent predictions
        actuals: Array of corresponding actual values
        threshold: Performance degradation threshold
    
    Returns:
        Boolean indicating if retraining is recommended
    """
    recent_mse = np.mean((predictions - actuals) ** 2)
    recent_dir_acc = np.mean((predictions > 0) == (actuals > 0))
    
    # Compare with original validation metrics
    original_mse = 0.000009  # From your final results
    degradation = recent_mse / original_mse
    
    if degradation > (1 + threshold) or recent_dir_acc < 0.45:
        print(f"Model performance degraded. Current MSE: {recent_mse:.8f}, Directional Accuracy: {recent_dir_acc:.2%}")
        return True
    
    return False

def main():
    # Load model
    print("Loading model...")
    model = load_trained_model()
    
    # Load historical data
    print("Loading historical data...")
    historical_data = load_historical_data()
    
    # Prepare scaler using historical data
    print("Preparing data...")
    _, scaler = prepare_data_for_prediction(historical_data)
    
    # Backtest to verify performance
    print("Running backtest...")
    backtest_results = backtest(model, historical_data, scaler)
    
    # Plot backtest results
    plt.figure(figsize=(14, 7))
    plt.plot(backtest_results['date'], backtest_results['capital'])
    plt.title('Model Backtest Performance')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.grid(True)
    plt.savefig('backtest_results.png')
    plt.show()
    
    # Calculate returns
    final_capital = backtest_results['capital'].iloc[-1]
    initial_capital = 10000
    total_return = (final_capital / initial_capital - 1) * 100
    
    print(f"Final portfolio value: ${final_capital:.2f}")
    print(f"Return: {total_return:.2f}%")
    
    # Additional analysis plots
    plt.figure(figsize=(14, 10))
    
    # Plot predicted vs actual returns
    plt.subplot(2, 1, 1)
    plt.scatter(backtest_results['actual_return'], backtest_results['predicted_return'], alpha=0.5)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Predicted vs Actual Returns')
    plt.grid(True)
    
    # Plot capital over time with buy/sell markers
    plt.subplot(2, 1, 2)
    plt.plot(backtest_results['date'], backtest_results['capital'])
    
    # Mark buy and sell signals
    buy_signals = backtest_results[backtest_results['signal'] == 'BUY']
    sell_signals = backtest_results[backtest_results['signal'] == 'SELL']
    
    plt.scatter(buy_signals['date'], buy_signals['capital'], marker='^', color='green', label='Buy')
    plt.scatter(sell_signals['date'], sell_signals['capital'], marker='v', color='red', label='Sell')
    
    plt.title('Capital over Time with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_analysis.png')
    plt.show()
    
    # If backtest looks good, start trading system
    if backtest_results['capital'].iloc[-1] > initial_capital:  # Only proceed if backtest is profitable
        print("Backtest shows positive returns")
        print("Starting trading system simulation...")
        trading_system(model, historical_data, scaler)
    else:
        print("Backtest shows negative returns. Model requires improvement before live trading.")

if __name__ == "__main__":
    main()