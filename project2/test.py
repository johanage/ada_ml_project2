# a script for tests and test functions
import numpy as np

def test_func_poly_deg_p(deg,avec, x):
    return  sum([avec[p]*x**p for p in range(deg+1)])

# for comparison with the example in the book: A high-variance low bias introudction to ML for physicists
def beales_func(x, y):
    return np.square(1.5 - x + x*y) + np.square(2.25 - x + x*y**2) + np.square(2.625 - x + x*y**3)
