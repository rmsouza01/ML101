import numpy as np


def dice_prob(auto_seg,gold_standard):
    return 2.0*(auto_seg*gold_standard).sum()/(auto_seg.sum()+gold_standard.sum()+1.0)

def dice(auto_seg,gold_standard, t = 0.5):
    auto_seg = auto_seg > t
    gold_standard = gold_standard > t
    intersec = np.logical_and(auto_seg,gold_standard)
    return 2.0*intersec.sum()/(auto_seg.sum()+gold_standard.sum())

def sensitivity(auto_seg,gold_standard):
    auto_seg = auto_seg > 0
    gold_standard = gold_standard > 0
    intersec = np.logical_and(auto_seg,gold_standard)
    return 1.0*intersec.sum()/(gold_standard.sum())

def specificity(auto_seg,gold_standard):
    auto_seg = auto_seg > 0
    gold_standard = gold_standard > 0
    intersec = np.logical_and(~auto_seg,~gold_standard)
    return 1.0*intersec.sum()/(gold_standard.size - gold_standard.sum())