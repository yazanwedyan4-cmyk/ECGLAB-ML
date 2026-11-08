"""Beat features: 12 baseline (4 RR + 8 beat-shape) and 15 (adds 3 RR features normalized by the patient's own rhythm)."""
import numpy as np
from scipy.stats import skew, kurtosis

FS, R0 = 360, 90          # R-peak index inside the 200-sample window

def qrs_width(b):
    base = np.median(b); half = 0.5 * abs(b[R0] - base); dev = np.abs(b - base) >= half
    l = r = R0
    while l > 0 and dev[l - 1]: l -= 1
    while r < len(b) - 1 and dev[r + 1]: r += 1
    return (r - l + 1) / FS

def beat_features(b):
    base = np.mean(b[R0 - 30:R0 - 15]); st = np.mean(b[R0 + 30:R0 + 45]) - base
    return [qrs_width(b), b[R0], st, b.std(), skew(b), kurtosis(b), b.max(), b.min()]

NAMES12 = ['pre_rr', 'post_rr', 'rr_ratio', 'local_rr', 'qrs_width', 'r_amp', 'st_dev', 'std', 'skew', 'kurt', 'max', 'min']
NAMES15 = NAMES12 + ['pre_rr_norm', 'post_rr_norm', 'local_rr_norm']

def build(X, RR):
    M = np.array([beat_features(b) for b in X], dtype='float32')
    F12 = np.hstack([RR[:, :4], M]); F15 = np.hstack([F12, RR[:, 4:7]])
    return F12, F15
