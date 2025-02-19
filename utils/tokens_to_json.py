import pandas as pd
import json

def extract_tokens_from_txt(file_path, output_json):
    """
    Reads a text file containing sequences of tokens, extracts the tokens,
    removes duplicates, and saves them to a JSON file.

    Args:
        file_path (str): Path to the input file.
        output_json (str): Path to the output JSON file.
    """
    try:
        df = pd.read_csv(file_path, header=None, names=["text"])

        tokens = df["text"].str.split(">").explode()
        tokens = tokens[tokens != ""]
        tokens = [token + ">" for token in tokens]
        unique_tokens = list(set(tokens))

        with open (output_json, "w") as f:
            json.dump(unique_tokens, f, indent = 4)
            
        print(f"Unique tokens saved to: {output_json}")

        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")