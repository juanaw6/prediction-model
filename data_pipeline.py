from utils.csv_converter import convert_csv_to_tokens, convert_csv_to_tokens_2
from utils.token_counter import count_tokens_from_txt
from utils.testset_maker import make_testset
from utils.tokens_to_json import extract_tokens_from_txt

# Setup
input_csv = "./data/raw/solusdt_5m_2021_2025.csv"
output_txt = "./data/train/sol_data.txt"
output_lined_txt = "./data/train/sol_lined.txt"
custom_tokens_json = "custom_tokens.json"
test_csv = "./data/test/testset.csv"

convert_csv_to_tokens(
    input_csv=input_csv,
    output_txt=output_txt,
    output_lined_txt=output_lined_txt,
    token_chunk_size=36
)

extract_tokens_from_txt(
    file_path=output_txt,
    output_json=custom_tokens_json
)

count_tokens_from_txt(output_txt)

make_testset(
    input_file=output_txt, 
    output_file=test_csv, 
    sequence_length=5, 
    max_samples=3000
)