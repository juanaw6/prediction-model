# Prediction Model

## Description
A machine learning framework for cryptocurrency price prediction, featuring:
- Multiple model architectures (GPT-2, LLaMA, TFT, HTN)
- Custom tokenizer for financial data
- Data pipeline for processing and tokenizing market data
- Backtesting framework for model evaluation

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/juanaw6/prediction-model.git
   ```
2. Navigate to the project directory:
   ```bash
   cd prediction-model
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Training a model
```bash
python models/gpt2/train_gpt2.py
```

### Running backtests
```bash
python scripts/backtest.py
```

### Data processing
```bash
python scripts/data_pipeline.py
```

## Directory Structure
```
matcher-model/
├── classes/                # Data fetchers and core classes
│   └── binance_data_fetcher.py
├── configs/                # Configuration files
│   └── bins_values.txt
├── custom-tokenizer/       # Custom tokenizer implementation
│   ├── custom_tokenizer.py
│   ├── custom_tokens.json
│   └── tokenizer_config.json
├── data/                   # Market data files
│   ├── raw/                # Raw CSV data
│   └── test/               # Test datasets
├── models/                 # Model implementations
│   ├── gpt2/               # GPT-2 models
│   │   ├── train_gpt2.py
│   │   ├── train_gpt2_2.py
│   │   └── test_inference_gpt2.py
│   ├── llama/              # LLaMA models
│   │   ├── train_llama.py
│   │   ├── train_llama_2.py
│   │   └── test_inference_llama.py
│   ├── htn.py              # Hierarchical Temporal Network
│   └── tft.py              # Temporal Fusion Transformer
├── scripts/                # Operational scripts
│   ├── backtest.py
│   ├── data_pipeline.py
│   ├── fetch_data.py
│   └── plot_changes.py
├── tests/                  # Test files
│   └── test_ft_model.py
└── utils/                  # Utility functions
    ├── csv_to_token_converter.py
    ├── dataset_splitter.py
    ├── token_counter.py
    └── token_extractor.py
```