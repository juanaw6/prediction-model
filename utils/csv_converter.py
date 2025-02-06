import pandas as pd

def convert_csv_to_tokens(input_csv, output_txt, output_lined_txt):
    """
    Converts price changes into token sequences and saves them to output files.

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_txt (str): Path to the output text file.
        output_lined_txt (str): Path to the output text file for the lined classification result.
    """
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    
    closes = df['close']
    opens = df['open']
    
    sequence = []
    
    UP_10_THRESHOLD = 1.0
    UP_9_THRESHOLD  = 0.9
    UP_8_THRESHOLD  = 0.8
    UP_7_THRESHOLD  = 0.7
    UP_6_THRESHOLD  = 0.6
    UP_5_THRESHOLD  = 0.5
    UP_4_THRESHOLD  = 0.4
    UP_3_THRESHOLD  = 0.3
    UP_2_THRESHOLD  = 0.2
    UP_1_THRESHOLD  = 0.1
    UP_0_THRESHOLD  = 0.05
    
    D_0_THRESHOLD   = -0.05 
    D_1_THRESHOLD   = -0.1
    D_2_THRESHOLD   = -0.2
    D_3_THRESHOLD   = -0.3
    D_4_THRESHOLD   = -0.4
    D_5_THRESHOLD   = -0.5
    D_6_THRESHOLD   = -0.6
    D_7_THRESHOLD   = -0.7
    D_8_THRESHOLD   = -0.8
    D_9_THRESHOLD   = -0.9
    D_10_THRESHOLD  = -1.0

    positive_thresholds = [
        (UP_10_THRESHOLD, '<U_10>'),
        (UP_9_THRESHOLD,  '<U_9>'),
        (UP_8_THRESHOLD,  '<U_8>'),
        (UP_7_THRESHOLD,  '<U_7>'),
        (UP_6_THRESHOLD,  '<U_6>'),
        (UP_5_THRESHOLD,  '<U_5>'),
        (UP_4_THRESHOLD,  '<U_4>'),
        (UP_3_THRESHOLD,  '<U_3>'),
        (UP_2_THRESHOLD,  '<U_2>'),
        (UP_1_THRESHOLD,  '<U_1>'),
        (UP_0_THRESHOLD,  '<U_0>')
    ]
    
    negative_thresholds = [
        (D_10_THRESHOLD, '<D_10>'),
        (D_9_THRESHOLD,  '<D_9>'),
        (D_8_THRESHOLD,  '<D_8>'),
        (D_7_THRESHOLD,  '<D_7>'),
        (D_6_THRESHOLD,  '<D_6>'),
        (D_5_THRESHOLD,  '<D_5>'),
        (D_4_THRESHOLD,  '<D_4>'),
        (D_3_THRESHOLD,  '<D_3>'),
        (D_2_THRESHOLD,  '<D_2>'),
        (D_1_THRESHOLD,  '<D_1>'),
        (D_0_THRESHOLD,  '<D_0>')
    ]

    for i in range(len(closes)):
        open_price = opens[i]
        close_price = closes[i]

        percent_change = ((close_price - open_price) / open_price) * 100
        
        if percent_change > UP_0_THRESHOLD:
            token = None
            for thresh, tok in positive_thresholds:
                if percent_change >= thresh:
                    token = tok
                    break
            if token is None:
                token = '<U_0>'
            sequence.append(token)
        
        elif percent_change < D_0_THRESHOLD:
            token = None
            for thresh, tok in negative_thresholds:
                if percent_change <= thresh:
                    token = tok
                    break
            if token is None:
                token = '<D_0>'
            sequence.append(token)
        
        else:
            sequence.append('<NEUTRAL>')

    result = ''.join(sequence)
    
    token_chunk_size = 36
    lined_result = '\n'.join(
        ''.join(sequence[i:i + token_chunk_size]) for i in range(0, len(sequence), token_chunk_size)
    )
    
    with open(output_txt, 'w') as f:
        f.write(result)
    print(f"Tokens saved to: {output_txt}")
    
    with open(output_lined_txt, 'w') as fl:
        fl.write(lined_result)
    print(f"Lined tokens saved to: {output_lined_txt}")