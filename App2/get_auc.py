import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

df = pd.read_csv("PREDS3_363.csv")
act = df["activity"].tolist()

act_true = df["tmp"].tolist()

print(len(act_true))
print(len(act))

score = roc_auc_score(act_true, act)

print("auc:", score)



#5824
#auc: 0.6705834777215096



#749
#749
#auc: 0.7069540229885057