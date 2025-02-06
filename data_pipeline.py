from utils.csv_converter import convert_csv_to_tokens
from utils.token_counter import count_tokens_from_txt
from utils.testset_maker import make_testset

# Setup
input_csv = "./data/raw/solusdt_5m_2021_2025.csv"
output_txt = "./data/train/sol_data.txt"
output_lined_txt = "./data/train/sol_lined.txt"
test_csv = "./data/test/testset.csv"

convert_csv_to_tokens(
    input_csv=input_csv,
    output_txt=output_txt,
    output_lined_txt=output_lined_txt
)

count_tokens_from_txt(output_txt)

make_testset(
    input_file=output_txt, 
    output_file=test_csv, 
    sequence_length=10, 
    max_samples=1000
)