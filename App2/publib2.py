import pubchempy as pcp
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

df = pd.read_csv("AID625185_table_sorted.csv")

datapoints = []
invalid = 0
total = 0

for index, row in df.iterrows():
    if index < 4:
        continue

    total += 1

    if (math.isnan(row['PUBCHEM_CID'])):
        print(index - 3, "INVALID")
        invalid += 1
        continue

    mol_cid = int(row['PUBCHEM_CID'])       # get SID from csv
    print(mol_cid, ":", index - 3)
    comp = pcp.Compound.from_cid(mol_cid)   # get substance from SID
    smiles = comp.canonical_smiles          # maybe use isomeric instead?

    activity = 0
    res = row['PUBCHEM_ACTIVITY_OUTCOME']
    if (res == "Active"):
        activity = 1

    print(smiles, "is", res, "aka", activity)
    datapoints.append([smiles, activity])

df2 = pd.DataFrame(datapoints, columns=["smiles", "activity"])
df2.to_csv('625185_tmp1.csv', index=False)

print("\n INVALID:", invalid)
print("\n TOTAL:", total)