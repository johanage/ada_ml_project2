# optimization script with all the optimization routines
import numpy as np
import autograd.numpy as np
# cost functions
def cost_OLS(X, y, theta):
    return np.sum((y-X @ theta)**2)

def cost_Ridge(X, y, theta, lmbda):
    return np.sum((y -  X @ theta + lmbda * np.eye(X.shape[1],X.shape[1]) )**2 )

def cost_Lasso(X, y, theta, lmbda):
    return np.sum( ( y- X @ beta )**2 ) + np.sum(np.abs(theta))

from autograd import grad
def grad_desc(X, y, cost_func, eta, idx_deriv=2, theta_init = None, check_conv = False, tol=1e-8, store_thetas = False, max_iter=int(1e4) ):
    # if init vals of theta is not given init theta with random normal distr vals
    if theta_init is None:
        theta = np.random.normal(size=(X.shape[1]))
    # remember to set correct idx_deriv for a cost func not defined with the same order of args as above
    train_grad = grad(cost_func, idx_deriv)
    # store the parameter for analysis
    if store_thetas:
        thetas = np.zeros((max_iter, theta.shape[0]))
    # init bconv
    bconv = False
    # start GD
    for niter in range(max_iter):
        # compute gradient with autograd
        gradients = train_grad(X, y, theta)
        # update theta by subtracting product of learning rate and gradients
        if check_conv:
            theta_old = theta.copy()
        theta -= eta*gradients
        # if user wants to check for convergence, when gradient disappears 
        if check_conv:
            del_theta = np.abs(theta - theta_old)
            # end iteration if cost is leq tol
            if max(del_theta) <= tol:
                bconv = True
                if store_thetas:
                    return thetas, bconv
                else:
                    return theta, bconv
        if store_thetas:
            thetas[niter] = theta
    # return either stored thetas or just last theta
    if store_thetas:
        return thetas, bconv
    else:
        return theta, bconv

def grad_desc_mom(X, y, cost_func, eta, delta, idx_deriv=2, theta_init = None, check_conv = False, tol=1e-8, store_thetas = False, max_iter=int(1e4) ):
    if min(delta) < 0 or max(delta) > 1:
        raise ValueError("delta must be between 0 and 1")
    return 0
