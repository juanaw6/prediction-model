import numpy as np
import pandas as pd

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
        temp_ranges = [[idx, r] for idx, r in enumerate(self.ranges)]
        temp_ranges = sorted(temp_ranges, key=lambda x: x[1].val)
        for i in range(len(temp_ranges) - 1):
            if temp_ranges[i][1].overlaps_with(temp_ranges[i+1][1]):
                r1, r2 = temp_ranges[i][1].split(temp_ranges[i+1][1])
                temp_ranges[i][1] = r1
                temp_ranges[i+1][1] = r2
        
        for r in temp_ranges:
            self.ranges[r[0]] = r[1]

    def get(self, point, default=-1):
        for range_ in self.ranges:
            if range_.min <= point <= range_.max:
                return self.last_occurrence[range_.val]
        return default

def get_sublists(lst, min_length=3, max_length=5):
    sublists = [lst[:i+min_length] for i in range(max_length - min_length + 1)]
    return sublists

def compute_tolerance(changes, factor=0.5):
    std_dev = np.std(changes)
    tolerance = std_dev * factor
    return tolerance

def within_tolerance(a, b, tolerance):
    return abs(a - b) <= tolerance

def pattern_matching_with_tolerance(text, pattern, dynamic_tolerance):
    m = len(pattern)
    n = len(text)
    bad_char_table = BadCharRangeTable()
    bad_char_table.generate(pattern, dynamic_tolerance)
    
    s = 0
    matches = []

    while s <= n - m:
        j = m - 1
        while j >= 0 and within_tolerance(text[s + j], pattern[j], dynamic_tolerance):
            j -= 1
        if j < 0:
            matches.append(s)
            s += 1
        else:
            bad_char_index = bad_char_table.get(text[s + j], default=-1)
            s += max(1, j - bad_char_index)

    return matches

def determine_action(changes):
    dynamic_tolerance = compute_tolerance(changes)
    patterns = get_sublists(changes)
    changes_to_match = changes[1:] # only match the pattern to the changes with the first data (so not always found one that is the pattern come from)
    
    matched = []
    for pattern in patterns:
        result = pattern_matching_with_tolerance(changes_to_match, pattern, dynamic_tolerance)
        result = [idx for idx in result if idx + len(pattern) < len(changes)]
        if not result:
            continue
        score = sum(1 if changes[idx] > 0 else -1 for idx in result)
        matched.append((pattern, result, score))

    return matched

# Load data
csv_file_path = "data_preprocessed.csv"
df = pd.read_csv(csv_file_path)
changes = df["changes"].tolist()
changes.reverse()

actions = determine_action(changes)

tolerance = compute_tolerance(changes)
print("Tolerance:", tolerance)

total_score = 0
for pattern, result, score in actions:
    print()
    print(f"Pattern {pattern}\n")
    print(f"At index: {result}")
    print(f"Found count: {len(result)}")
    print(f"Score: {score}")
    print()
    total_score += score

# Final decision
print(f"Total Score: {total_score}")
if total_score > 0:
    print("Decision: BUY")
elif total_score < 0:
    print("Decision: SELL")
else:
    print("Decision: None")