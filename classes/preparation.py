import pandas as pd
from joblib import Parallel, delayed

from classes.bm import BoyerMoore

class Preparation:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Map 'BULLISH' to 'U', 'NEUTRAL' to 'N', and 'BEARISH' to 'D'
        df['changes_label'] = df['changes_label'].map({'BULLISH': 'U', 'NEUTRAL': 'N', 'BEARISH': 'D'})
        df['next_changes_label'] = df['next_changes_label'].map({'BULLISH': 'U', 'NEUTRAL': 'N', 'BEARISH': 'D'})

        for i in range(1, 51):
            df[f'changes_label_-{i}'] = df['changes_label'].shift(i)
        
        df['target_class'] = df['next_changes_label']
        
        # Dropping unnecessary columns
        df.drop(['timestamp', 'open', 'high', 'low', 'close', 'changes', 'volume', 'next_changes_label'], axis=1, inplace=True)
        
        df.dropna(axis=0, inplace=True)
        
        # df = self.find_pattern_score(df)
        # Saving the transformed DataFrame to a CSV file
        df.to_csv('test.csv', index=False)
        return df

    def find_pattern_score(self, df: pd.DataFrame) -> pd.DataFrame:
        bm = BoyerMoore()
        num_jobs = -1  # Use all available CPU cores
        
        def _calculate_score_static(df, pattern_len, idx_range):
            scores = []
            text = ''.join(df['changes_label'].fillna('').tolist())
            for idx in idx_range:
                if idx < pattern_len:
                    scores.append(0)
                    continue
                pattern = text[idx - pattern_len:idx]
                matches = bm.search_pattern(text[:idx], pattern)
                score = 0
                for match_idx in matches:
                    if match_idx + pattern_len < len(df):
                        next_label = df['changes_label'].iloc[match_idx + pattern_len]
                        if next_label == 'U':
                            score += 1
                        elif next_label == 'D':
                            score -= 1
                scores.append(score)
            return scores

        for pattern_len in range(3, 11):
            indices = range(len(df))
            chunk_size = max(1, len(df) // Parallel(n_jobs=num_jobs)._effective_n_jobs())
            index_ranges = [list(indices[i:i + chunk_size]) for i in range(0, len(df), chunk_size)]

            # Parallelize score computation
            results = Parallel(n_jobs=num_jobs)(
                delayed(_calculate_score_static)(df, pattern_len, idx_range) for idx_range in index_ranges
            )

            # Flatten results and assign to the DataFrame
            scores = [score for result in results for score in result]
            df[f'pattern_len_{pattern_len}_score'] = scores
        return df

    @staticmethod
    def _get_suffixes_static(lst, min_length=3, max_length=3):
        """Static method for generating suffixes."""
        return [lst[i:] for i in range(len(lst) - min_length, -1, -1) if max_length is None or len(lst[i:]) <= max_length]