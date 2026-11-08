"""Raw MIT-BIH -> filtered beats + RR features + record id + split label (from splits json)."""
import json, numpy as np, wfdb
from scipy.signal import butter, sosfiltfilt
import protocol
from protocol import AAMI, CLASSES, MITDB

FS, PRE, POST = 360, 90, 110            # beat window: 90 samples before the R-peak, 110 after

def bandpass(x, fs=FS, lo=0.5, hi=40, order=3):
    sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x)           # zero-phase

def preprocess(x, fs=FS):
    x = bandpass(x, fs)
    return (x - x.mean()) / (x.std() + 1e-8)     # z-score per record

def lead_index(r):
    names = [n.upper() for n in r.sig_name]
    return names.index('MLII') if 'MLII' in names else 0

def process_record(rec):
    r = wfdb.rdrecord(f'{MITDB}/{rec}'); a = wfdb.rdann(f'{MITDB}/{rec}', 'atr')
    sig = preprocess(r.p_signal[:, lead_index(r)], r.fs)
    keep = [(s, AAMI[sym]) for s, sym in zip(a.sample, a.symbol) if sym in AAMI]
    pos = np.array([s for s, _ in keep]); lab = [l for _, l in keep]
    rr_all = np.diff(pos); B, RR, Y = [], [], []
    for i in range(1, len(pos) - 1):
        if pos[i] - PRE < 0 or pos[i] + POST > len(sig):
            continue
        pre_rr, post_rr = rr_all[i - 1], rr_all[i]
        local = np.mean(rr_all[max(0, i - 10):i])                 # previous intervals only
        hist = rr_all[max(0, i - 21):i - 1]
        med = np.median(hist) if len(hist) else pre_rr
        RR.append([pre_rr / r.fs, post_rr / r.fs, pre_rr / post_rr, local / r.fs,
                   pre_rr / med, post_rr / med, local / med])
        B.append(sig[pos[i] - PRE:pos[i] + POST]); Y.append(CLASSES.index(lab[i]))
    return np.array(B, dtype='float32'), np.array(RR, dtype='float32'), np.array(Y)

def build_all(out_npz):
    split = {int(k): v for k, v in json.load(open(protocol.SPLIT_FILE)).items()}
    X, RR, Y, REC, SPL = [], [], [], [], []
    for rec in protocol.all_records():
        B, R, y = process_record(rec)
        X.append(B); RR.append(R); Y.append(y)
        REC.append(np.full(len(y), rec)); SPL.append(np.full(len(y), split[rec]))
    out = dict(X=np.concatenate(X), RR=np.concatenate(RR), y=np.concatenate(Y),
               rec=np.concatenate(REC), split=np.concatenate(SPL))
    np.savez_compressed(out_npz, **out)
    return out
