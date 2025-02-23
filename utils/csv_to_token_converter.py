import pandas as pd

def convert_csv_to_tokens(input_csv_path, output_txt_path, num_classes):
    """
    Converts price changes into token sequences and saves them to output files.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_txt_path (str): Path to the output text file.
        output_lined_txt (str): Path to the output text file for the lined classification result.
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
    