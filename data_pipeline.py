from utils.csv_to_token_converter import convert_csv_to_tokens
from utils.token_counter import count_tokens_in_file
from utils.dataset_splitter import split_dataset_into_train_test
from utils.token_extractor import extract_tokens_to_json

# Setup
input_csv_path = "./data/raw/solusdt_5m_2021_2025.csv"
output_token_sequence_file = "./data/train/sol_token_sequences.txt"
output_lined_token_file = "./data/train/sol_lined_tokens.txt"
custom_tokens_json_path = "custom_tokens.json"
test_dataset_csv_path = "./data/test/test_dataset.csv"

convert_csv_to_tokens(
    input_csv_path=input_csv_path,
    output_txt_path=output_token_sequence_file,
    num_classes=5
)


extract_tokens_to_json(
    token_sequence_path=output_token_sequence_file,
    output_json=custom_tokens_json_path
)

count_tokens_in_file(output_token_sequence_file)

split_dataset_into_train_test(
    input_path=output_token_sequence_file,
    train_output_path=output_lined_token_file,
    test_output_path=test_dataset_csv_path,
    train_token_context=36,
    test_token_context=36,
    max_samples=3000,
    windowed=True,
    train_window_step=36,
    test_window_step=1
)