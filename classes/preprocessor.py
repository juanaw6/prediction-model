import pandas as pd

class Preprocessor:
    def __init__(
        self,
        high_threshold: float = 0.05,
        low_threshold: float = -0.05
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def _get_changes_label(self, change: float) -> str:
        """Determine the label for the given change value."""
        if change > self.high_threshold:
            return 'BULLISH'
        elif self.low_threshold <= change <= self.high_threshold:
            return 'NEUTRAL'
        elif change < self.low_threshold:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def transform(
        self,
        df: pd.DataFrame,
        output_filepath: str = None
    ) -> pd.DataFrame:
        if 'changes' not in df.columns:
            raise ValueError("Dataframe must contain a 'changes' column.")
        
        df['changes_label'] = df['changes'].apply(self._get_changes_label)
        df['next_changes_label'] = df['changes_label'].shift(-1)
        
        if output_filepath is not None:
            df.to_csv(output_filepath, index=False)
        
        return df
    