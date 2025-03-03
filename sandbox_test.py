import pandas as pd

class Example():
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
class Backtest():
    def __init__(self, data_csv_path):
        self.data: pd.DataFrame = pd.read_csv(data_csv_path)
        self.data['change'] = ((self.data['close'] - self.data['open']) / self.data["open"]) * 100
        
    def run(self, num_example, data_len=None):
        changes = self.data['change'].values[:10].tolist()
        examples = []
        
        for i in range(num_example):
            if i + data_len >= len(changes):
                print("[DEBUG] Not enough data to generate more example!")
                break
            data = changes[i:i+data_len]
            target = changes[i+data_len]
            examples.append(Example(data, target))
        
        print(changes)
        for ex in examples:
            print(ex.data, ex.target)
            
if __name__ == "__main__":
    bt = Backtest("./data/raw/solusdt_5m_2024_2025.csv")
    bt.run(num_example=6, data_len=5)
    