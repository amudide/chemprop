import pubchempy as pcp
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv("AID363_table.csv")

datapoints = []

for index, row in df.iterrows():
    if index < 3:
        continue

    mol_sid = int(row['PUBCHEM_SID'])       # get SID from csv
    print(mol_sid, ":", index - 2)         
    sub = pcp.Substance.from_sid(mol_sid)   # get substance from SID
    comp = sub.standardized_compound        # get compound from substance
    smiles = comp.canonical_smiles          # maybe use isomeric instead?

    activity = 0
    res = row['PUBCHEM_ACTIVITY_OUTCOME']
    if (res == "Active"):
        activity = 1

    print(smiles, "is", activity)
    datapoints.append([smiles, activity])

df2 = pd.DataFrame(datapoints, columns=["smiles", "activity"])
df2.to_csv('363.csv', index=False)