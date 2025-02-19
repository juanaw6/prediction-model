import os
import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

def create_and_save_tokenizer(
    custom_tokens=None,
    output_file="custom_tokenizer.json",
    output_dir="./custom-tokenizer"
):
    """
    Creates a custom tokenizer with special tokens, sets the pad token to "<PAD>",
    and saves it permanently, ensuring the directory exists.

    Args:
        custom_tokens (list, optional): List of custom tokens. Defaults to predefined ones.
        output_file (str, optional): Filename to save the tokenizer JSON. Defaults to "custom_tokenizer.json".
        output_dir (str, optional): Directory to save the tokenizer. Defaults to "./custom-tokenizer".
    """

    if custom_tokens is None:
        custom_tokens = []

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))

    special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    all_tokens = special_tokens + custom_tokens

    vocab = {token: idx for idx, token in enumerate(all_tokens)}

    tokenizer.add_tokens(list(vocab.keys()))
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer_path = os.path.join(output_dir, output_file)
    tokenizer.save(tokenizer_path)
    
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    fast_tokenizer.add_special_tokens({
        "pad_token": "<PAD>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "unk_token": "<UNK>",
    })

    fast_tokenizer.save_pretrained(output_dir)

    print(f"Tokenizer created and saved to '{tokenizer_path}'")
    print(f"Tokenizer directory: '{output_dir}'")
    print(f"Vocabulary size: {len(fast_tokenizer)}")
    print("\nVocabulary:")
    for token, idx in vocab.items():
        print(f"{token}: {idx}")

if __name__ == "__main__":
    json_file_path = "custom_tokens.json"
    try:
        with open(json_file_path, "r") as f:
            custom_tokens = json.load(f)
        print(f"Loaded custom tokens from {json_file_path}")
    except FileNotFoundError:
        print(f"Error: Custom tokens JSON file not found at {json_file_path}. Using an empty list instead.")
        custom_tokens = []

    create_and_save_tokenizer(custom_tokens=custom_tokens)