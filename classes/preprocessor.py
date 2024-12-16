import pandas as pd

class Preprocessor:
    def __init__(
        self,
        high_threshold: float = 0.05,
        low_threshold: float = -0.05
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
    def transform(
        self,
        df: pd.DataFrame,
        output_filepath: str = None
    ) -> pd.DataFrame:
        if 'changes' not in df.columns:
            raise ValueError("Dataframe must contain a 'changes' column.")
        
        target_class = []
        
        for i in range(len(df) - 1):
            next_change = df['changes'].iloc[i + 1]
            
            if next_change > self.high_threshold:
                target_class.append('BUY')
            elif self.low_threshold <= next_change <= 0.05:
                target_class.append('HOLD')
            elif next_change < self.low_threshold:
                target_class.append('SELL')
        
        target_class.append(None)
        df['target'] = target_class
        
        if output_filepath != None:
            df.to_csv(output_filepath, index=False)
        
        return df