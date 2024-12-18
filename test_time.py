def boyer_moore_pattern_matching(lst, pattern):
    n, m = len(lst), len(pattern)
    bad_char = {ch: m - i - 1 for i, ch in enumerate(pattern)}
    i = 0
    matches = []
    
    while i <= n - m:
        j = m - 1
        while j >= 0 and lst[i + j] == pattern[j]:
            j -= 1
        if j < 0:
            matches.append(i)
            i += m  # Skip the matched part (because of the bad_char rule)
        else:
            i += bad_char.get(lst[i + j], m)  # Skip based on the bad character
    return matches

def sliding_window_python(lst, pattern, window_size):
    matches = []
    for i in range(len(lst) - window_size + 1):
        window = lst[i:i + window_size]
        if window == pattern:
            matches.append(i)
    return matches

import numpy as np

def sliding_window_numpy(lst, pattern, window_size):
    arr = np.array(lst)
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window_size)
    matches = [i for i, window in enumerate(windows) if np.array_equal(window, pattern)]
    return matches

import time

# Test data
test_list = list(range(100000))  # Large list
pattern = [500, 501, 502]  # Small pattern
window_size = len(pattern)

# Function to time each approach
def time_boyer_moore():
    start = time.time()
    boyer_moore_pattern_matching(test_list, pattern)
    end = time.time()
    return end - start
boyer_moore_time = time_boyer_moore()
print(f"Boyer-Moore Time: {boyer_moore_time:.6f} seconds")

def time_sliding_window_python():
    start = time.time()
    print(sliding_window_python(test_list, pattern, window_size))
    end = time.time()
    return end - start
sliding_window_python_time = time_sliding_window_python()
print(f"Sliding Window (Python) Time: {sliding_window_python_time:.6f} seconds")

# def time_sliding_window_numpy():
#     start = time.time()
#     sliding_window_numpy(test_list, pattern, window_size)
#     end = time.time()
#     return end - start
# sliding_window_numpy_time = time_sliding_window_numpy()
# print(f"Sliding Window (NumPy) Time: {sliding_window_numpy_time:.6f} seconds")