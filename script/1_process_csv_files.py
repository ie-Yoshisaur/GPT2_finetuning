import pandas as pd
from glob import glob

path = './csv/unprocessed_csv/*.csv'
csv_files = glob(path)
merged_df = None 

for csv_file in csv_files:
    if merged_df is None:
        merged_df = pd.read_csv(csv_file)
    else:
        df_to_merge = pd.read_csv(csv_file)
        merged_df = pd.concat([merged_df, df_to_merge]).reset_index(drop=True)

merged_df = merged_df.rename(columns={'artificial_line': 'input', 'reply': 'output'})

proccessed_csv_path = './csv/processed_csv/input_output.csv'

merged_df.to_csv(proccessed_csv_path)

with open(proccessed_csv_path) as f:
    content = f.read()

while ',,' in content:
    content = content.replace(',,', ',')

content = content.replace(",\n", "\n")

with open(proccessed_csv_path, mode='w') as f:
    f.write(content)
