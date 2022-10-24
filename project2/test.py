# a script for tests and test functions

def test_func_poly_deg_p(deg,avec, x):
    return  sum([avec[p]*x**p for p in range(deg+1)])
