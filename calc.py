from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import pandas as pd

experiment = "all_b64"
file_name = f"/home/ipeas22062020/Downloads/{experiment}/content/output/test_logits_test_deberta_Base.txt"
test_data = "data/test.tsv"

df = pd.read_csv(test_data, sep="\t")
print(df.head())

true_data = df.sentiment.values.astype(int)
pred_data = np.empty(true_data.shape).astype(int)

max_laptop = 638
lap_true_data = df.sentiment.values.astype(int)[:max_laptop]
lap_pred_data = np.empty(lap_true_data.shape).astype(int)

rest_true_data = df.sentiment.values.astype(int)[max_laptop:]
rest_pred_data = np.empty(rest_true_data.shape).astype(int)

with open(file_name, "r") as f_:
    i_line = 0
    data = f_.read()
    print("Total samples: ", len(data.split('\n')) - 1)
    print(df.shape)
    for i, line in enumerate(data.split("\n")): 
        if len(line) < 1:
            continue

        pred = [float(x) for x in line.split('\t')]

        if i_line < max_laptop:
            lap_pred_data[i_line] = np.argmax(pred) - 1
        else:
            rest_pred_data[i_line - max_laptop] = np.argmax(pred) - 1

        i_line += 1

print("Laptop results: ")
cf_matrix = confusion_matrix(lap_true_data, lap_pred_data)
acc_score = accuracy_score(lap_true_data, lap_pred_data)
f_score = f1_score(lap_true_data, lap_pred_data, average="macro")

print(cf_matrix)
print(f"acc:\t\t{acc_score}")
print(f"f_score:\t{f_score}")

print("Restaurant results: ")

cf_matrix = confusion_matrix(rest_true_data, rest_pred_data)
acc_score = accuracy_score(rest_true_data, rest_pred_data)
f_score = f1_score(rest_true_data, rest_pred_data, average="macro")

print(cf_matrix)
print(f"acc:\t\t{acc_score}")
print(f"f_score:\t{f_score}")