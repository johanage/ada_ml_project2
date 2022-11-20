# all the metrics used in project2

import numpy as np

def MSE(y, ytilde):
    return np.sum((y-ytilde)**2)/y.shape[0]

def R2score(y, ytilde):
    ssres = np.sum((y - ytilde)**2)
    sstot = np.sum((y-np.sum(y)/y.shape[0])**2)
    return 1 - ssres/sstot

