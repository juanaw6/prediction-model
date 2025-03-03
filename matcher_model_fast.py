import numpy as np
import pandas as pd
import time
import multiprocessing as mp
from functools import partial

class Range():
    def __init__(self, minimum, val, maximum):
        self.min = minimum
        self.val = val
        self.max = maximum

    def __repr__(self):
        return f"Range({self.min}, {self.val}, {self.max})"

    def overlaps_with(self, other):
        return not (self.max < other.min or self.min > other.max)

    def split(self, other):
        midpoint = (self.val + other.val) / 2
        if self.val < other.val:
            return Range(self.min, self.val, midpoint), Range(midpoint, other.val, other.max)
        else:
            return Range(other.min, other.val, midpoint), Range(midpoint, self.val, self.max)

class BadCharRangeTable():
    def __init__(self):
        self.ranges = []
        self.last_occurrence = {}

    def generate(self, pattern, tolerance):
        self.ranges = []
        self.last_occurrence = {}
        for i, char in enumerate(pattern):
            if char not in self.last_occurrence:
                min_val = char - tolerance
                max_val = char + tolerance
                self.ranges.append(Range(min_val, char, max_val))
            self.last_occurrence[char] = i
        self.split_all_overlapping_ranges()

    def split_all_overlapping_ranges(self):
        if not self.ranges:
            return
            
        idx_arr = np.arange(len(self.ranges))
        val_arr = np.array([r.val for r in self.ranges])
        sorted_indices = np.argsort(val_arr)
        
        temp_ranges = [(idx_arr[i], self.ranges[i]) for i in sorted_indices]
        
        for i in range(len(temp_ranges) - 1):
            if temp_ranges[i][1].overlaps_with(temp_ranges[i+1][1]):
                r1, r2 = temp_ranges[i][1].split(temp_ranges[i+1][1])
                temp_ranges[i] = (temp_ranges[i][0], r1)
                temp_ranges[i+1] = (temp_ranges[i+1][0], r2)
        
        for idx, r in temp_ranges:
            self.ranges[idx] = r

    def get(self, point, default=-1):
        for range_ in self.ranges:
            if range_.min <= point <= range_.max:
                return self.last_occurrence[range_.val]
        return default

def compute_tolerance(changes, factor=0.3):
    std_dev = np.std(changes)
    return std_dev * factor

def pattern_matching_with_tolerance(text_array, pattern_array, dynamic_tolerance):
    m = len(pattern_array)
    n = len(text_array)
    
    if m > n:
        return []
    
    bad_char_table = BadCharRangeTable()
    bad_char_table.generate(pattern_array, dynamic_tolerance)
    
    s = 0
    matches = []
    
    diffs = np.empty(m, dtype=np.float64)
    
    while s <= n - m:
        diffs = np.abs(text_array[s:s+m] - pattern_array)
        
        if np.all(diffs <= dynamic_tolerance):
            matches.append(s)
            s += 1
            continue
            
        j = m - 1
        while j >= 0 and diffs[j] <= dynamic_tolerance:
            j -= 1
                
        if j < 0:
            matches.append(s)
            s += 1 
        else:
            bad_char_index = bad_char_table.get(text_array[s + j], default=-1)
            s += max(1, j - bad_char_index)

    return matches

class PatternFound():
    __slots__ = ['pattern', 'idx_found', 'score']
    
    def __init__(self, pattern, idx_found, score):
        self.pattern = pattern
        self.idx_found = idx_found
        self.score = score
    
    def __repr__(self):
        return (
                f"\n[DEBUG] Pattern (len: {len(self.pattern)}) found at index {self.idx_found}\n"
                f"[DEBUG] Score: {self.score}\n")
        
class Result():
    __slots__ = ['total_score', 'patterns_found']
    
    def __init__(self, total_score, patterns_found):
        self.total_score = total_score
        self.patterns_found = patterns_found

def process_pattern(changes_array, pattern_data, dynamic_tolerance, n):
    start_idx, pattern = pattern_data
    result = pattern_matching_with_tolerance(changes_array, pattern, dynamic_tolerance)
    idx_found = [idx for idx in result if idx + len(pattern) < n and idx != start_idx]
    
    if not idx_found:
        return None
    
    next_indices = np.array([idx + len(pattern) for idx in idx_found], dtype=np.int32)
    next_values = changes_array[next_indices]
    score = np.sum(np.where(next_values > 0, 1, -1))
    
    return PatternFound(pattern, idx_found, score)

def calculate_score(changes):
    changes_array = np.array(changes, dtype=np.float64)
    dynamic_tolerance = compute_tolerance(changes_array)
    
    n = len(changes_array)
    min_length = 3
    patterns = []
    
    for i in range(n - min_length, -1, -1):
        if n - i <= n // 2:
            patterns.append((i, changes_array[i:].copy()))
    
    num_cores = mp.cpu_count()
    with mp.Pool(processes=num_cores) as pool:
        process_pattern_partial = partial(process_pattern, changes_array, 
                                         dynamic_tolerance=dynamic_tolerance, n=n)
        results = pool.map(process_pattern_partial, patterns)
    
    matched = [r for r in results if r is not None]
    total_score = sum(pattern.score for pattern in matched)

    return Result(total_score, matched)

if __name__ == "__main__":
    csv_file_path = "./data/raw/solusdt_5m_2024_2025.csv"
    df = pd.read_csv(csv_file_path)
    
    df['change'] = ((df['close'] - df['open']) / df["open"]) * 100
    changes = df['change'].values[:5000].tolist()

    print("[DEBUG] Start")
    start_time = time.time()
    result = calculate_score(changes)
    duration = time.time() - start_time
    print(f"[DEBUG] Duration: {duration * 1000:.4f} ms")

    tolerance = compute_tolerance(changes)
    print("[DEBUG] Tolerance:", tolerance)

    for pattern_found in result.patterns_found:
        print(pattern_found)

    print(f"[DEBUG] Total Score: {result.total_score}")