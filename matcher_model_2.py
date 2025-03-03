import numpy as np
import pandas as pd
import time
from collections import defaultdict

class Range:
    def __init__(self, minimum, val, maximum):
        self.min = minimum
        self.val = val
        self.max = maximum

    def __repr__(self):
        return f"Range({self.min:.4f}, {self.val:.4f}, {self.max:.4f})"

    def overlaps_with(self, other):
        return not (self.max < other.min or self.min > other.max)

    def split(self, other):
        midpoint = (self.val + other.val) / 2
        if self.val < other.val:
            return Range(self.min, self.val, midpoint), Range(midpoint, other.val, other.max)
        else:
            return Range(other.min, other.val, midpoint), Range(midpoint, self.val, self.max)

class BadCharRangeTable:
    def __init__(self):
        self.ranges = []
        self.last_occurrence = {}

    def generate(self, pattern, tolerance):
        self.ranges = []
        self.last_occurrence = {}
        for i, char in enumerate(pattern):
            min_val = char - tolerance
            max_val = char + tolerance
            self.ranges.append(Range(min_val, char, max_val))
            self.last_occurrence[char] = i
        self.split_all_overlapping_ranges()

    def split_all_overlapping_ranges(self):
        if not self.ranges:
            return
            
        # Sort ranges by value for better overlap detection
        temp_ranges = [(idx, r) for idx, r in enumerate(self.ranges)]
        temp_ranges.sort(key=lambda x: x[1].val)
        
        i = 0
        while i < len(temp_ranges) - 1:
            if temp_ranges[i][1].overlaps_with(temp_ranges[i+1][1]):
                r1, r2 = temp_ranges[i][1].split(temp_ranges[i+1][1])
                temp_ranges[i] = (temp_ranges[i][0], r1)
                temp_ranges[i+1] = (temp_ranges[i+1][0], r2)
                # Rescan from the beginning in case the split created new overlaps
                i = 0
            else:
                i += 1
        
        for idx, r in temp_ranges:
            self.ranges[idx] = r

    def get(self, point, default=-1):
        for range_ in self.ranges:
            if range_.min <= point <= range_.max:
                return self.last_occurrence[range_.val]
        return default

def compute_adaptive_tolerance(changes, window_size=100, factor=0.3):
    """Calculate adaptive tolerance based on local volatility"""
    if len(changes) <= window_size:
        return np.std(changes) * factor
    
    tolerances = []
    for i in range(0, len(changes), window_size//2):
        window = changes[i:i+window_size]
        if len(window) >= window_size//2:
            tolerances.append(np.std(window) * factor)
    
    return np.mean(tolerances)

def rolling_zscore(series, window=20):
    """Calculate rolling z-scores to normalize the changes"""
    rolling_mean = pd.Series(series).rolling(window=window).mean()
    rolling_std = pd.Series(series).rolling(window=window).std()
    
    # Handle first window elements
    rolling_mean.iloc[:window-1] = rolling_mean.iloc[window-1]
    rolling_std.iloc[:window-1] = rolling_std.iloc[window-1]
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.mean(rolling_std[rolling_std > 0]))
    
    z_scores = (series - rolling_mean) / rolling_std
    return z_scores.tolist()

def pattern_matching_with_tolerance(text, pattern, tolerance, min_confidence=0.7):
    m = len(pattern)
    n = len(text)
    
    if m > n:
        return []
    
    bad_char_table = BadCharRangeTable()
    bad_char_table.generate(pattern, tolerance)
    
    s = 0
    matches = []
    
    while s <= n - m:
        j = m - 1
        match_count = 0
        
        while j >= 0:
            if abs(text[s + j] - pattern[j]) <= tolerance:
                match_count += 1
            j -= 1
        
        # Calculate match confidence
        confidence = match_count / m
        if confidence >= min_confidence:
            matches.append((s, confidence))
            s += 1
        else:
            # Use the bad character heuristic for faster skipping
            s += max(1, m // 4)  # More aggressive skip for efficiency

    return matches

class PatternFound:
    def __init__(self, pattern, idx_found, confidences, score, next_values):
        self.pattern = pattern
        self.idx_found = idx_found
        self.confidences = confidences
        self.score = score
        self.next_values = next_values
        self.pattern_length = len(pattern)
    
    def __repr__(self):
        avg_confidence = sum(self.confidences) / len(self.confidences) if self.confidences else 0
        return (
            f"\n[DEBUG] Pattern (len: {self.pattern_length}) found at {len(self.idx_found)} locations\n"
            f"[DEBUG] Average match confidence: {avg_confidence:.2f}\n"
            f"[DEBUG] Score: {self.score}\n"
            f"[DEBUG] Next values prediction: {'UP' if self.score > 0 else 'DOWN'}, strength: {abs(self.score)}"
        )
        
class Result:
    def __init__(self, total_score, patterns_found):
        self.total_score = total_score
        self.patterns_found = patterns_found
        self.patterns_found.sort(key=lambda x: abs(x.score), reverse=True)  # Sort by score magnitude

def calculate_significance(pattern_length, occurrences, total_length):
    """Calculate pattern significance based on length and frequency"""
    return (pattern_length * occurrences) / (total_length * 0.1)

def calculate_score(changes, normalize=True):
    if normalize:
        # Normalize the changes using rolling z-scores
        changes = rolling_zscore(changes)
    
    dynamic_tolerance = compute_adaptive_tolerance(changes)
    
    n = len(changes)
    # Increase min_length for more meaningful patterns
    min_length = 5  # Minimum pattern length
    max_length = min(50, n // 3)  # Maximum pattern length
    
    patterns = []
    # Evaluate patterns of different lengths
    for length in range(min_length, max_length + 1, 2):
        for i in range(0, n - length + 1, length // 2):
            patterns.append((i, changes[i:i+length]))
    
    matched = []
    total_score = 0
    pattern_cache = {}  # Cache to avoid recomputing the same patterns

    for start_idx, pattern in patterns:
        # Create a hash of the pattern to avoid duplicate computations
        pattern_key = tuple(np.round(pattern, 3))
        
        if pattern_key in pattern_cache:
            continue
        pattern_cache[pattern_key] = True
        
        result = pattern_matching_with_tolerance(changes, pattern, dynamic_tolerance)
        
        # Filter out the pattern's own start position and ensure there's space for a next value
        valid_matches = [(idx, conf) for idx, conf in result 
                         if idx != start_idx and idx + len(pattern) < n]
        
        if len(valid_matches) < 2:  # Need at least 2 matches to be significant
            continue
        
        idx_found = [idx for idx, _ in valid_matches]
        confidences = [conf for _, conf in valid_matches]
        
        next_indices = [idx + len(pattern) for idx in idx_found]
        next_values = [changes[idx] for idx in next_indices if idx < n]
        
        if not next_values:
            continue
            
        # Calculate a weighted score based on next values and match quality
        weighted_next_values = [next_val * conf for next_val, conf in zip(next_values, confidences)]
        score = sum(weighted_next_values)
        
        # Add significance factor based on pattern length and frequency
        significance = calculate_significance(len(pattern), len(idx_found), n)
        adjusted_score = score * significance
        
        pattern_result = PatternFound(pattern, idx_found, confidences, adjusted_score, next_values)
        matched.append(pattern_result)
        total_score += adjusted_score

    # Filter out low significance patterns
    matched = [p for p in matched if abs(p.score) > 1.0]
    
    return Result(total_score, matched)

if __name__ == "__main__":
    csv_file_path = "./data/raw/solusdt_5m_2024_2025.csv"
    df = pd.read_csv(csv_file_path)
    
    df['change'] = ((df['close'] - df['open']) / df["open"]) * 100
    changes = df['change'].values[:1000].tolist()

    print("[DEBUG] Start")
    start_time = time.time()
    result = calculate_score(changes)
    duration = time.time() - start_time
    print(f"[DEBUG] Duration: {duration * 1000:.4f} ms")

    tolerance = compute_adaptive_tolerance(changes)
    print("[DEBUG] Adaptive Tolerance:", tolerance)
    print(f"[DEBUG] Found {len(result.patterns_found)} significant patterns")

    # Show top 10 patterns by score magnitude
    for i, pattern_found in enumerate(result.patterns_found[:10]):
        print(f"[PATTERN {i+1}]{pattern_found}")

    print(f"[DEBUG] Total Score: {result.total_score}")
    print(f"[DEBUG] Prediction: {'BULLISH' if result.total_score > 0 else 'BEARISH'}, Strength: {abs(result.total_score):.2f}")