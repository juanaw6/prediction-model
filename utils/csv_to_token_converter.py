import numpy as np
import pandas as pd

def convert_csv_to_tokens(input_csv_path, output_txt_path, num_classes):
    """
    Converts price changes into token sequences and saves them to output files.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_txt_path (str): Path to the output text file.
    """
    df = pd.read_csv(input_csv_path, parse_dates=['timestamp'])
    
    df['percent_change'] = (df['close'] - df['open']) / df['open'] * 100

    if num_classes % 2 == 0:
        print("Even class value. Results may be incorrect, use odd value.")
    class_midpoint = num_classes // 2
    class_labels = [str(i - class_midpoint) for i in range(num_classes)]
    df['class'] = pd.qcut(df['percent_change'], q=num_classes, labels=class_labels)
    
    quantile_vals = df['percent_change'].quantile([i/num_classes for i in range(1, num_classes)])

    quantile_values_output_path = "quantile_values.txt"
    quantile_vals.to_csv(quantile_values_output_path, header=['Quantile Value'], index=False)
    print(f"Quantile values saved to {quantile_values_output_path}")
    
    sequence = []
    
    classes = df['class']
    
    for cl in classes:
        current_class = int(cl)
        if current_class < 0:
            sequence.append(f"<DOWN_{abs(current_class)}>")
        elif current_class > 0:
            sequence.append(f"<UP_{current_class}>")
        else:
            sequence.append("<NEUTRAL>")
    
    result = ''.join(sequence)
    
    with open(output_txt_path, 'w') as f:
        f.write(result)
    print(f"Tokens saved to: {output_txt_path}")
    
def convert_csv_to_tokens_more_bins(input_csv_path, output_txt_path=None):
    """
    Converts price changes into token sequences and saves them to output file.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_txt_path (str): Path to the output text file.
    """
    
    df = pd.read_csv(input_csv_path, parse_dates=['timestamp'])
    
    df['percent_change'] = (df['close'] - df['open']) / df['open'] * 100
    
    # Check for NaN values before cutting
    if df['percent_change'].isna().any():
        print("Warning: NaN values found in percent_change column")


    quantile_cut_classes = 5
        
    quantile_vals = list(df['percent_change'].quantile([i/quantile_cut_classes for i in range(1, quantile_cut_classes)]))
    max_val = np.max(df['percent_change'])
    min_val = np.min(df['percent_change'])
    
    new_quantile_vals = quantile_vals.copy()
    k=1
    for i in range(len(quantile_vals)-1):   
        mid_point = (quantile_vals[i] + quantile_vals[i+1]) / 2
        new_quantile_vals.insert(k, mid_point)
        k += 2
    
    bins = [min_val] + new_quantile_vals + [max_val]

    print(bins)        
    
    class_midpoint = (len(bins) - 1) // 2
    class_labels = [int(i - class_midpoint) for i in range(len(bins) - 1)]
    
    print(class_labels)
    df['class'] = pd.cut(df['percent_change'], bins=bins, labels=class_labels)
    
    bins_values_output_path = "bins_values.txt"
    pd.DataFrame(bins, columns=['Bins Value']).to_csv(bins_values_output_path, index=False)
    print(f"Bins values saved to {bins_values_output_path}")
    
    sequence = []
    
    classes = df['class']
    
    for cl in classes:
        if pd.isna(cl):
            sequence.append("<NEUTRAL>")
            continue
            
        current_class = int(cl)
        if current_class < 0:
            sequence.append(f"<DOWN_{abs(current_class)}>")
        elif current_class > 0:
            sequence.append(f"<UP_{current_class}>")
        else:
            sequence.append("<NEUTRAL>")
    
    result = ''.join(sequence)
    
    if output_txt_path:
        with open(output_txt_path, 'w') as f:
            f.write(result)
        print(f"Tokens saved to: {output_txt_path}")
    
    return result

# if __name__ == "__main__":
#     test_convert_csv_to_tokens(input_csv_path=r'data\raw\solusdt_5m_2021_2025.csv')