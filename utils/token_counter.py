import pandas as pd

def count_tokens_in_file(file_path):
    """
    Reads a text file containing sequences of tokens, counts occurrences of each token, 
    and prints a sorted list of tokens and their counts.

    Args:
        file_path (str): Path to the input file.
    """
    df = pd.read_csv(file_path, header=None, names=["text"])

    tokens = df["text"].str.split(">").explode()
    tokens = tokens[tokens != ""]

    tokens = tokens + ">"

    token_counts = tokens.value_counts().reset_index()
    token_counts.columns = ["Token", "Count"]

    print(token_counts.to_string(index=False))