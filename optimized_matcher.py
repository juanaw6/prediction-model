import numpy as np
import pandas as pd

def load_and_preprocess(csv_file_path):
    df = pd.read_csv(csv_file_path)
    changes = ((df["close"] - df["open"]) / df["open"]) * 100
    return changes.values  # Convert to NumPy array for vectorization

def compute_tolerance(changes, factor=0.3):
    std_dev = np.std(changes)
    return std_dev * factor

def pattern_matching_with_tolerance(text, pattern, tolerance):
    m = len(pattern)
    n = len(text)
    if m == 0 or m > n:
        return []
    
    # Create a 2D array where each row is a sliding window of text
    windows = np.lib.stride_tricks.sliding_window_view(text, window_shape=m)
    
    # Compute the absolute differences and check within tolerance
    diffs = np.abs(windows - pattern)
    within_tol = np.all(diffs <= tolerance, axis=1)
    
    # Find indices where the pattern matches
    match_indices = np.where(within_tol)[0]
    return match_indices.tolist()

def get_patterns(changes, min_length=3, max_length=10):
    patterns = []
    n = len(changes)
    for length in range(max_length, min_length - 1, -1):
        for i in range(n - length + 1):
            pattern = changes[i:i+length]
            patterns.append(pattern)
    return patterns

def determine_action(changes, tolerance, min_length=3, max_length=10):
    patterns = get_patterns(changes, min_length, max_length)
    matched = []
    total_score = 0
    
    # Precompute future changes for scoring
    future_changes = changes[max_length:]
    
    for pattern in patterns:
        indices = pattern_matching_with_tolerance(changes, pattern, tolerance)
        # Ensure that there is a future change to evaluate the score
        valid_indices = [idx for idx in indices if idx + len(pattern) < len(changes)]
        if not valid_indices:
            continue
        # Vectorized scoring
        scores = np.where(changes[np.array(valid_indices) + len(pattern)] > 0, 1, -1)
        score = np.sum(scores)
        matched.append((pattern, valid_indices, score))
        total_score += score
    
    return matched, total_score

def main():
    # Load and preprocess data
    csv_file_path = "futures_data.csv"
    changes = load_and_preprocess(csv_file_path)
    
    # Compute tolerance
    tolerance = compute_tolerance(changes)
    print("Tolerance:", tolerance)
    
    # Determine actions
    matched_patterns, total_score = determine_action(changes, tolerance)
    
    # Display matched patterns and scores
    # for pattern, indices, score in matched_patterns:
    #     print("-----------------------------------------------------------")
    #     print(f"Pattern {pattern}\nFound at indices {indices}, Score: {score}")
    #     print("-----------------------------------------------------------")
    
    # Final decision
    print(total_score)
    if total_score > 0:
        print("Decision: BUY")
    elif total_score < 0:
        print("Decision: SELL")
    else:
        print("Decision: None")

if __name__ == "__main__":
    main()