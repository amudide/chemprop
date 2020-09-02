import pubchempy as pcp
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

df = pd.read_csv("AID765821_more.csv")

datapoints = []
invalid = 0
total = 0

used = {}

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

    print(smiles, "is", res, "aka", activity)
    datapoints.append([smiles, activity])

df2 = pd.DataFrame(datapoints, columns=["smiles", "activity"])
df2.to_csv('765821_more.csv', index=False)

print("\n INVALID:", invalid)
print("\n TOTAL:", total)