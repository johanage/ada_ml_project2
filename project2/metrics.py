# all the metrics used in project2

import numpy as np

def MSE(y, ytilde):
    return np.sum((y-ytilde)**2)/y.shape[0]

def R2score():
    return 0

