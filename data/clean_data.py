import glob
import pandas as pd
import os

output_dir = "output"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for csv_data in glob.glob("*.csv"):
    df = pd.read_csv(csv_data)
    df_clean = df[df["sentiment"] != 0]
    df_clean.sentiment = df_clean.sentiment.apply(lambda x: 0 if x == -1 else x)
    # df_clean[df_clean["sentiment"] == -1] = 0
    df_clean.to_csv(f"{output_dir}/{csv_data.split('.')[0]}_clean.tsv", sep='\t', index=False)