import pubchempy as pcp
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

df = pd.read_csv("preprocess/src-more.csv")

datapoints = []
invalid = 0
total = 0

used = {}

usedS = {}
df3 = pd.read_csv("363.csv")
for index, row in df3.iterrows():
    usedS[row['smiles']] = True

for index, row in df.iterrows():
    if index < 0:
        continue

    total += 1

    if (math.isnan(row['cid'])):
        print(index + 1, "INVALID")
        invalid += 1
        continue

    mol_cid = int(row['cid'])       # get SID from csv

    if mol_cid in used:
        print("Already there")
        continue

    used[mol_cid] = True

    print(mol_cid, ":", index + 1)
    comp = pcp.Compound.from_cid(mol_cid)   # get substance from SID
    smiles = comp.canonical_smiles          # maybe use isomeric instead?

    activity = 0
    res = row['activity']
    if (res == "Active"):
        activity = 1

    if (smiles in usedS):
        print("Duplicate from 363.")
        continue

    print(smiles, "is", res, "aka", activity)
    datapoints.append([smiles, activity])

df2 = pd.DataFrame(datapoints, columns=["smiles", "activity"])
df2.to_csv('363_test.csv', index=False)

print("\n INVALID:", invalid)
print("\n TOTAL:", total)