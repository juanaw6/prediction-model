import numpy as np
import pandas as pd
import ahocorasick

def get_prefixes(lst, min_length=3, max_length=10):
    prefixes = [lst[:i+min_length] for i in range(max_length - min_length + 1)]
    return prefixes

def pattern_matching(text, patterns):
    A = ahocorasick.Automaton()
    for idx, pattern in enumerate(patterns):
        pattern_str = "".join(pattern)  # Join the list elements to form a string
        A.add_word(pattern_str, idx)
        
    A.make_automaton()
    
    matches = []
    text_str = "".join(text)  # Convert text list into a single string
    for end_index, pattern_id in A.iter(text_str):
        matches.append((end_index, patterns[pattern_id]))
        
    return matches

# Function to determine the action based on patterns and matching results
def determine_action(changes):
    # Generate patterns (prefixes of varying lengths)
    patterns = get_prefixes(changes, min_length=3, max_length=10)
    
    # Match the patterns in the 'changes' list (excluding the first element)
    matches = pattern_matching(changes, patterns)
    
    # Store the matches
    total_score = 0
    match_counts = {}
    
    for match in matches:
        end_idx, pattern = match
        pattern_length = len(pattern)
        
        # Count how many times each pattern is matched
        if pattern not in match_counts:
            match_counts[pattern] = 0
        match_counts[pattern] += 1
        
        # Calculate score based on whether the change after the match is positive or negative
        change_at_match = changes[end_idx + 1] if end_idx + 1 < len(changes) else 0
        if change_at_match > 0:
            total_score += 1  # BUY signal
        elif change_at_match < 0:
            total_score -= 1  # SELL signal
    
    # Print match counts and results
    for pattern, count in match_counts.items():
        print(f"Pattern {pattern} found {count} times")
    
    return total_score

# Load data (assuming the 'changes_label' column exists in the CSV)
csv_file_path = "data_preprocessed.csv"
df = pd.read_csv(csv_file_path)
changes = df["changes_label"].tolist()
changes.reverse()  # Assuming you want to reverse the changes list (if required)

# Determine the action and calculate the total score
total_score = determine_action(changes)

# Final decision
print(f"Total Score: {total_score}")
if total_score > 0:
    print("Decision: BUY")
elif total_score < 0:
    print("Decision: SELL")
else:
    print("Decision: None")