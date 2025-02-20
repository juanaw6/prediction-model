import pandas as pd
import numpy as np
np.NaN = np.nan
import pandas_ta as ta

def to_lined(sequence, token_chunk_size, output_lined_txt):
    """
    Saves a sequence of tokens to a text file, chunking them into lines.

    Args:
        sequence (list): List of tokens.
        token_chunk_size (int): Number of tokens per line.
        output_lined_txt (str): Path to the output text file.
    """
    lined_result = '\n'.join(
        ''.join(sequence[i:i + token_chunk_size]) for i in range(0, len(sequence), token_chunk_size)
    )
    with open(output_lined_txt, 'w') as fl:
        fl.write(lined_result)
    print(f"Lined tokens saved to: {output_lined_txt}")


def convert_csv_to_tokens(input_csv, output_txt, output_lined_txt, token_chunk_size):
    """
    Converts price changes into token sequences and saves them to output files.

    Args:
        input_csv (str): Path to the input CSV file.
        output_txt (str): Path to the output text file.
        output_lined_txt (str): Path to the output text file for the lined classification result.
        token_chunk_size (int): Number of tokens per line for the lined output.
    """
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    
    df['percent_change'] = (df['close'] - df['open']) / df['open'] * 100

    num_classes = 5

    if num_classes % 2 == 0:
        print("Even class value. Results may be incorrect, use odd value.")
    class_midpoint = num_classes // 2
    class_labels = [str(i - class_midpoint) for i in range(num_classes)]
    df['class'] = pd.qcut(df['percent_change'], q=num_classes, labels=class_labels)
    
    quantile_vals = df['percent_change'].quantile([i/num_classes for i in range(1, num_classes)])
    # print("\nQuantile Values:")
    # print(list(quantile_vals))

    output_txt_file = "quantile_values.txt"
    quantile_vals.to_csv(output_txt_file, header=['Quantile Value'], index=False)
    print(f"Quantile values saved to {output_txt_file}")
    
    sequence = []
    
    classes = df['class']
    
    for cl in classes:
        current_class = int(cl)
        if current_class < 0:
            sequence.append(f"<DOWN_{current_class}>")
        elif current_class > 0:
            sequence.append(f"<UP_{current_class}>")
        else:
            sequence.append("<NEUTRAL>")
    
    result = ''.join(sequence)
    
    with open(output_txt, 'w') as f:
        f.write(result)
    print(f"Tokens saved to: {output_txt}")
    
    to_lined(sequence=sequence, token_chunk_size=token_chunk_size, output_lined_txt=output_lined_txt)
    
    
def convert_csv_to_tokens_2(input_csv, output_txt, output_lined_txt, token_chunk_size):
    """
    Converts the CSV file by first calculating MACD and RSI (using pandas-ta) and then
    generating tokens that combine the price-action (bullish/bearish), MACD and RSI levels.
    For example, if for the current candle the close is above the open (bullish), MACD > 0, and RSI > 50,
    the token will be "<U_MABOVE0_RSIABOVE50>". For bearish candles or other indicator levels, the token is built accordingly.

    code

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_txt (str): Path to the output text file.
        output_lined_txt (str): Path to the output text file for the lined tokens.
        token_chunk_size (int): Number of tokens per line in the lined output.
    """
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])

    macd_df = ta.macd(close=df['close'])
    df['macd'] = macd_df['MACD_12_26_9']
    df['rsi'] = ta.rsi(close=df['close'])
    
    df.dropna(inplace=True)

    tokens = []
    
    std = np.std(((df['close'] - df['open']) / df['open']) * 100)
    print(f"std Change: {std}")

    for _, row in df.iterrows():
        if row['close'] > row['open']:
            direction = "U"
        elif row['close'] < row['open']:
            direction = "D"
        else:
            direction = "N"
        
        if row['macd'] > 0:
            macd_token = "M-ABOVE-0"
        else:
            macd_token = "M-BELOW-0"

        if row['rsi'] >=70:
            rsi_token = "RSI-OB"
        elif row['rsi'] <= 30:
            rsi_token = "RSI-OS"
        else:
            rsi_token = "RSI-N"
        
        token = f"<{direction}_{macd_token}_{rsi_token}>"
        tokens.append(token)

    result = ''.join(tokens)
    with open(output_txt, 'w') as f:
        f.write(result)
    print(f"Tokens saved to: {output_txt}")
    
    to_lined(sequence=tokens, token_chunk_size=token_chunk_size, output_lined_txt=output_lined_txt)
    