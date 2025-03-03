from matcher_model_fast import calculate_score
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
        cumm_reward = 0
        
        for example in tqdm(examples, desc="Backtesting"):
            result = calculate_score(example.data)
            if result.total_score == 0:
                no_action +=  1
                continue
            score = 1 if result.total_score > 0 else -1
            target_signed = 1 if example.target > 0 else -1
            if score == target_signed:
                correct_pred += 1
                cumm_reward += abs(example.target)
            else:
                cumm_reward -= abs(example.target)
        
        accuracy = (correct_pred + no_action) / len(examples) * 100
        accuracy_real = correct_pred / len(examples) * 100
        print(f"Accuracy with no action: {accuracy:.4f}%")
        print(f"Accuracy without no action: {accuracy_real:.4f}%")
        print(f"Cummulative PnL: {cumm_reward}%")        
        
if __name__ == "__main__":
    data = "./data/raw/solusdt_5m_2024_2025.csv"
    bt = Backtest(data)
    bt.run(num_example=300, data_len=5000)
