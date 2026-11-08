"""Project protocol: MIT-BIH only, patient-wise split 70/15/15 decided from raw annotations (before any training)."""
import os, json, numpy as np, pandas as pd, wfdb

MITDB = '/content/drive/MyDrive/ecglab/data/mitdb'
SPLIT_FILE = '/content/drive/MyDrive/ecglab/data/split_v2_70_15_15.json'
PACED = [102, 104, 107, 217]                     # paced records are excluded
SAME_PERSON = {202: 201}                          # records 201 and 202 are the same person
CLASSES = ['N', 'S', 'V', 'F', 'Q']
AAMI = {**dict.fromkeys(['N', 'L', 'R', 'e', 'j'], 'N'), **dict.fromkeys(['A', 'a', 'J', 'S'], 'S'),
        **dict.fromkeys(['V', 'E'], 'V'), 'F': 'F', **dict.fromkeys(['/', 'f', 'Q'], 'Q')}
SHARE = {'train': .70, 'val': .15, 'test': .15}

def all_records():
    recs = sorted(int(f[:-4]) for f in os.listdir(MITDB) if f.endswith('.hea') and f[:-4].isdigit())
    return [r for r in recs if r not in PACED]

def beat_counts():
    rows = {}
    for r in all_records():
        a = wfdb.rdann(f'{MITDB}/{r}', 'atr'); c = np.zeros(5, int)
        for s in a.symbol:
            if s in AAMI: c[CLASSES.index(AAMI[s])] += 1
        rows[r] = c
    df = pd.DataFrame(rows).T; df.columns = CLASSES
    return df

def make_split():
    """Deterministic greedy allocation of persons (not beats) to train/val/test, matching class shares."""
    cnt = beat_counts(); cnt['person'] = [SAME_PERSON.get(r, r) for r in cnt.index]
    g = cnt.groupby('person')[['N', 'S', 'V', 'F']].sum(); tot = g.sum()
    names = list(SHARE); share = np.array(list(SHARE.values()))
    alloc, n, assign = np.zeros((3, 4)), np.zeros(3), {}
    order = (g[['S', 'V', 'F']] / tot[['S', 'V', 'F']]).sum(1).sort_values(ascending=False).index
    for k in order:                               # persons rich in rare classes first
        costs = []
        for i in range(3):
            a2, n2 = alloc.copy(), n.copy(); a2[i] += g.loc[k].values; n2[i] += 1
            costs.append((((a2 - share[:, None] * tot.values) / tot.values) ** 2).sum()
                         + (((n2 - share * len(g)) / len(g)) ** 2).sum())
        i = int(np.argmin(costs)); assign[int(k)] = names[i]; alloc[i] += g.loc[k].values; n[i] += 1
    return assign, g, alloc, n

def record_split(assign):
    return {int(r): assign[SAME_PERSON.get(r, r)] for r in all_records()}
