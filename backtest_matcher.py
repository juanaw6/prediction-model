import numpy as np
from matcher_model_2 import calculate_score
import pandas as pd
from tqdm import tqdm

class Example():
    def __init__(self, data, target):
        self.data = data
        self.target = target

class Backtest():
    def __init__(self, data_csv_path):
        self.data: pd.DataFrame = pd.read_csv(data_csv_path)
        self.data['change'] = ((self.data['close'] - self.data['open']) / self.data["open"]) * 100
        
    def run(self, num_example=1, data_len=100):
        changes = self.data['change'].tolist()
        examples = []
        
        for i in range(num_example):
            if i + data_len >= len(changes):
                print("[DEBUG] Not enough data to generate more example!")
                break
            data = changes[i:i+data_len]
            target = changes[i+data_len]
            examples.append(Example(data, target))
            
        correct_pred = 0
        neutral = 0
        cumm_pnl = 0
        
        # scores = []
        for example in tqdm(examples, desc="[RUNNING] Backtesting"):
            result = calculate_score(example.data)
            if result.total_score == 0:
                neutral +=  1
                continue
            action = 1 if result.total_score > 0 else -1
            target_signed = 1 if example.target > 0 else -1
            if action == target_signed:
                correct_pred += 1
                cumm_pnl += abs(example.target)
            else:
                cumm_pnl -= abs(example.target)
        #     scores.append(abs(result.total_score))
        
        # print(f"Average score: {np.mean(scores)}")
        
        total_example = len(examples)
        accuracy_neutral = (correct_pred + neutral) / total_example * 100
        real_accuracy = correct_pred / (total_example - neutral) * 100 if (total_example - neutral) > 0 else 0
        
        # print(f"Data length: {data_len}, Examples: {num_example}")
        # print(f"Accuracy with no action: {accuracy:.4f}%")
        # print(f"Accuracy without no action: {real_accuracy:.4f}%")
        # print(f"Cummulative PnL: {cumm_pnl:.4f}%")
        
        return {
            "accuracy_with_neutral": accuracy_neutral,
            "real_accuracy": real_accuracy,
            "cumm_pnl": cumm_pnl,
            "data_len": data_len,
            "num_example": total_example
        }
            
if __name__ == "__main__":
    data = "./data/raw/solusdt_15m_2024_2025.csv"
    
    results = []
    
    data_lengths = [3000]

    example_counts = [10000]
    
    for data_len in data_lengths:
        for num_example in example_counts:
            print(f"\n[DEBUG] Running with data_len={data_len}, num_example={num_example}")
            bt = Backtest(data)
            result = bt.run(num_example=num_example, data_len=data_len)
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    for col in ['accuracy_with_neutral', 'real_accuracy', 'cumm_pnl']:
        results_df[col] = results_df[col].map('{:.2f}%'.format)
        
    results_df = results_df.sort_values(['data_len', 'num_example'])
    
    print("\n[DEBUG] Results")
    print(results_df.to_string(index=False))
    
    print("\n[DEBUG] Best Results by Real Accuracy")
    best_df = pd.DataFrame(results).sort_values('real_accuracy', ascending=False)
    for col in ['accuracy_with_neutral', 'real_accuracy', 'cumm_pnl']:
        best_df[col] = best_df[col].map('{:.2f}%'.format)
    print(best_df.head(5).to_string(index=False))
