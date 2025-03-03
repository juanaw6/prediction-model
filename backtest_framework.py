from matcher_model import calculate_score
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
        
    def run(self, num_example, data_len=None):
        if data_len != None:
            changes = self.data['change'].values[:data_len].tolist()
        else:
            changes = self.data['change'].tolist()
        
        itr = [changes[:i] for i in range(len(changes), len(changes) - num_example, -1)]
        examples = []
        for it in itr:
            target = it[-1]
            data = it[:-1]
            examples.append(Example(data, target))
            
        correct_pred = 0
        no_action = 0
        cumm_pnl = 0
        
        for example in tqdm(examples, desc="Backtesting"):
            result = calculate_score(example.data)
            if result.total_score == 0:
                no_action +=  1
                continue
            score = 1 if result.total_score > 0 else -1
            target_signed = 1 if example.target > 0 else -1
            if score == target_signed:
                correct_pred += 1
                cumm_pnl += abs(example.target)
            else:
                cumm_pnl -= abs(example.target)
        
        accuracy = (correct_pred + no_action) / len(examples) * 100
        accuracy_real = correct_pred / (len(examples) - no_action) * 100 if (len(examples) - no_action) > 0 else 0
        
        # print(f"Data length: {data_len}, Examples: {num_example}")
        # print(f"Accuracy with no action: {accuracy:.4f}%")
        # print(f"Accuracy without no action: {accuracy_real:.4f}%")
        # print(f"Cummulative PnL: {cumm_pnl:.4f}%")
        
        return {
            "accuracy_with_no_action": accuracy,
            "accuracy_without_no_action": accuracy_real,
            "cumm_pnl": cumm_pnl,
            "data_len": data_len,
            "num_example": num_example
        }
            
if __name__ == "__main__":
    data = "./data/raw/solusdt_5m_2024_2025.csv"
    
    # Test different combinations of parameters
    results = []
    
    # Test with different data lengths
    data_lengths = [150, 200, 500, 1000, 2000, 3000]
    
    # Test with different numbers of examples
    example_counts = [100, 200, 300, 500]
    
    for data_len in data_lengths:
        for num_example in example_counts:
            if data_len is None or num_example < data_len:
                print(f"\n--- Running with data_len={data_len}, num_example={num_example} ---")
                bt = Backtest(data)
                result = bt.run(num_example=num_example, data_len=data_len)
                results.append(result)
    
    # Display the best results
    print("\n=== Best Results by Accuracy (with no action) ===")
    best_by_accuracy = sorted(results, key=lambda x: x["accuracy_with_no_action"], reverse=True)[:5]
    for res in best_by_accuracy:
        print(f"data_len={res['data_len']}, num_example={res['num_example']}, accuracy={res['accuracy_with_no_action']:.2f}%, real_accuracy={res['accuracy_without_no_action']:.2f}%, pnl={res['cumm_pnl']:.2f}%")
    
    print("\n=== Best Results by Accuracy (excluding no action) ===")
    best_by_real_accuracy = sorted(results, key=lambda x: x["accuracy_without_no_action"], reverse=True)[:5]
    for res in best_by_real_accuracy:
        print(f"data_len={res['data_len']}, num_example={res['num_example']}, accuracy={res['accuracy_with_no_action']:.2f}%, real_accuracy={res['accuracy_without_no_action']:.2f}%, pnl={res['cumm_pnl']:.2f}%")
    
    print("\n=== Best Results by PnL ===")
    best_by_pnl = sorted(results, key=lambda x: x["cumm_pnl"], reverse=True)[:5]
    for res in best_by_pnl:
        print(f"data_len={res['data_len']}, num_example={res['num_example']}, accuracy={res['accuracy_with_no_action']:.2f}%, real_accuracy={res['accuracy_without_no_action']:.2f}%, pnl={res['cumm_pnl']:.2f}%")
