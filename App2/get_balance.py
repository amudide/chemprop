import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv("625185_full.csv")
act = df["activity"].tolist()

print("Percentage of positives:", sum(act) / len(act) * 100)