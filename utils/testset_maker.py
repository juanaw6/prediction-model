import os
import pandas as pd

def make_testset(input_file, output_file, sequence_length=36, max_samples=None):
    """
    Converts a sequence of tokens into a CSV file where each row contains an input sequence
    and its corresponding target token using pandas.

    Parameters:
        input_file (str): Path to the input text file containing the sequence of tokens.
        output_file (str): Path to the output CSV file.
        sequence_length (int, optional): Number of tokens in each input sequence. Default is 36.
        max_samples (int, optional): Maximum number of test samples to generate. If None, uses all available data.
    """
    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            sequence = f.read().strip()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Split the sequence by '>' and reattach the '>' to each token (ignoring empty tokens)
    tokens = [token + '>' for token in sequence.split('>') if token]
    total_tokens = len(tokens)
    
    if total_tokens <= sequence_length:
        print("Not enough tokens in the input file to create any sample.")
        return

    # Generate input-target pairs
    data = []
    for i in range(total_tokens - sequence_length):
        input_seq = ''.join(tokens[i:i + sequence_length])
        target = tokens[i + sequence_length]
        data.append([input_seq, target])
        if max_samples is not None and len(data) >= max_samples:
            break

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=["input_sequence", "target"])

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write the DataFrame to CSV
    try:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"CSV file '{output_file}' created successfully with {len(df)} samples.")
    except Exception as e:
        print(f"Error writing output file: {e}")