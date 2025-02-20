import pandas as pd
import pandas_ta as ta

def transform_csv(input_file, output_file):
    """
    Reads a CSV file with columns: timestamp, open, high, low, close, volume,
    computes RSI (14-period) and MACD (default: fast=12, slow=26, signal=9) from the 'close' price,
    and writes a new CSV containing only the columns: close, volume, rsi, macd.
    
    Parameters:
    - input_file: Path to the input CSV file.
    - output_file: Path to the output CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Ensure required columns are present
    for col in ['close', 'volume']:
        if col not in df.columns:
            raise KeyError(f"Input CSV must contain a '{col}' column.")
    
    # Calculate RSI with a 14-period window
    df['rsi'] = ta.rsi(close=df['close'], length=14)
    
    # Calculate MACD. This returns a DataFrame with three columns:
    # 'MACD_12_26_9', 'MACDh_12_26_9', and 'MACDs_12_26_9'.
    macd_df = ta.macd(close=df['close'])
    
    # For this script, we'll use the MACD line (not the histogram or signal)
    df['macd'] = macd_df['MACD_12_26_9']
    
    # Create an output DataFrame with the selected columns
    output_df = df[['close', 'volume', 'rsi', 'macd']]
    output_df.dropna(inplace=True)
    
    # Write the output DataFrame to a new CSV file
    output_df.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}")

# Example usage: adjust the file paths as needed.
if __name__ == '__main__':
    input_csv_path = r'data\raw\solusdt_5m_2024_2025.csv'
    output_csv_path = 'rm_output.csv'
    transform_csv(input_csv_path, output_csv_path)