import glob
import pandas as pd
import os

output_dir = "output"
train_split = 0.8

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for csv_data in glob.glob("*.csv"):
    df = pd.read_csv(csv_data)
    # Previous used positive and negative labels
    # df_clean = df[df["sentiment"] != 0]
    # df_clean.sentiment = df_clean.sentiment.apply(lambda x: 0 if x == -1 else x)
    # df_clean[df_clean["sentiment"] == -1] = 0

    if "train" in csv_data:
        df_train = df.sample(frac=train_split)
        df_test = df.drop(df_train.index)

        df_train.to_csv(f"{output_dir}/{csv_data.split('.')[0]}.tsv", sep='\t', index=False)
        test_filename = csv_data.replace("train", "dev")
        df_test.to_csv(f"{output_dir}/{test_filename.split('.')[0]}.tsv", sep='\t', index=False)
    else:
        df.to_csv(f"{output_dir}/{csv_data.split('.')[0]}.tsv", sep='\t', index=False)
