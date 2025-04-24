import pandas as pd

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df.to_csv(output_path, index=False)


