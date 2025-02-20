import os
import pandas as pd

def split_dataset_into_train_test(
    input_path, 
    train_output_path, 
    test_output_path, 
    train_token_context, 
    test_token_context, 
    max_samples=None, 
    windowed=False, 
    train_window_step=1,
    test_window_step=1,
):
    """
    Splits a sequence of tokens into training and testing datasets.

    Parameters:
        input_path (str): Path to the input text file containing the sequence of tokens.
        train_output_path (str): Path to save the training data.
        test_output_path (str): Path to save the testing data.
        train_token_context (int): Number of tokens in each training sequence.
        test_token_context (int): Number of tokens in each testing sequence.
        max_samples (int, optional): Maximum number of test samples to generate. If None, uses all available data.
        windowed (bool, optional): Whether to use windowed processing. Default is False.
        window_step (int, optional): Number of tokens to slide the window by. Default is 1.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            sequence = f.read().strip()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    tokens = [token + '>' for token in sequence.split('>') if token]
    total_tokens = len(tokens)
    
    if total_tokens <= (train_token_context + test_token_context):
        print("Not enough tokens in the input file to create any sample.")
        return
    
    split_point = int(total_tokens * 0.9)
    train_tokens = tokens[:split_point]
    test_tokens = tokens[split_point:]

    test_data = []
    if windowed:
        step = min(test_window_step, test_token_context)
    else:
        step = test_token_context

    for i in range(0, len(test_tokens) - test_token_context):
        if windowed:
            if i % step != 0:
                continue
        input_seq = ''.join(test_tokens[i:i + test_token_context])
        target = test_tokens[i + test_token_context]
        test_data.append([input_seq, target])
        if max_samples is not None and len(test_data) >= max_samples:
            break

    df = pd.DataFrame(test_data, columns=["input_sequence", "target"])

    for path in [train_output_path, test_output_path]:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    try:
        df.to_csv(test_output_path, index=False, encoding='utf-8')
        print(f"CSV file '{test_output_path}' created successfully with {len(df)} samples.")
    except Exception as e:
        print(f"Error writing test output file: {e}")
        
    save_lined_tokens(
        sequence=train_tokens, 
        token_chunk_size=train_token_context, 
        output_lined_txt=train_output_path, 
        windowed=windowed, 
        window_step=train_window_step
    )
        
def save_lined_tokens(sequence, token_chunk_size, output_lined_txt, windowed=False, window_step=1):
    """
    Saves a sequence of tokens to a text file, chunking them into lines.

    Args:
        sequence (list): List of tokens.
        token_chunk_size (int): Number of tokens per line.
        output_lined_txt (str): Path to the output text file.
        windowed (bool, optional): Whether to use windowed processing. Default is False.
        window_step (int, optional): Number of tokens to slide the window by. Default is 1.
    """
    data = []
    if windowed:
        step = min(window_step, token_chunk_size)
    else:
        step = token_chunk_size

    for i in range(0, len(sequence) - token_chunk_size):
        if windowed:
            if i % step != 0:
                continue
        input_seq = ''.join(sequence[i:i + token_chunk_size])
        data.append(input_seq)
    
    lined_result = '\n'.join(data)

    try:
        with open(output_lined_txt, 'w', encoding='utf-8') as fl:
            fl.write(lined_result)
        print(f"Lined tokens saved to: {output_lined_txt} with {len(data)} sequences")
    except Exception as e:
        print(f"Error writing training output file: {e}")