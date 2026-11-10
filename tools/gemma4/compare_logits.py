#!/usr/bin/env python3
"""compare_logits.py A.bin B.bin N_VOCAB: float32 logits files (rows of N_VOCAB), position by position: the largest difference, the RMS
difference over the spread of the logits, and whether the best token and the top five agree."""
import sys

import numpy as np

a = np.fromfile(sys.argv[1], dtype=np.float32)
b = np.fromfile(sys.argv[2], dtype=np.float32)
nv = int(sys.argv[3])
assert a.size == b.size and a.size % nv == 0, (a.size, b.size, nv)
a = a.reshape(-1, nv)
b = b.reshape(-1, nv)
worst = 0.0
for i in range(len(a)):
    d = np.abs(a[i] - b[i])
    spread = a[i].std()
    rms = np.sqrt(np.mean((a[i] - b[i]) ** 2)) / spread
    top1 = a[i].argmax() == b[i].argmax()
    t5 = len(set(np.argsort(-a[i])[:5]) & set(np.argsort(-b[i])[:5]))
    print("pos %2d: max|d| %.2e  rms/spread %.2e  top1 %s  top5 overlap %d/5" % (i, d.max(), rms, "same" if top1 else "DIFF", t5))
    worst = max(worst, rms)
print("worst rms/spread: %.3e" % worst)
