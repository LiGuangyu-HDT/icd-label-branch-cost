"""Report the distribution of synonym lengths per ICD code.

Reads the multi-synonym mapping produced by the preprocessing step and prints the mean and
median number of words per description. These are the figures behind the paper's
"1 to 30 words, mean 5.61" and the 35,684 x 30 = 1,070,520 padded-slot count.

Usage, from model/main (where embedding/ lives):
    python embedding/summary.py
"""
import json

import numpy as np

with open('embedding/icd_mimic3_random_sort.json', 'r') as f:
    descriptions = json.load(f)

counts = np.array([len(d) for d in descriptions])
print("descriptions : %d" % len(counts))
print("words/desc   : mean %.2f | median %.0f | min %d | max %d"
      % (counts.mean(), np.median(counts), counts.min(), counts.max()))
print("padded slots : %d (= %d x 30)" % (len(counts) * 30, len(counts)))
print("real tokens  : %d" % counts.sum())
print("redundancy   : %.2fx" % (len(counts) * 30 / counts.sum()))
