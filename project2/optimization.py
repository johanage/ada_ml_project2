# optimization script with all the optimization routines
import numpy as np
import autograd.numpy as np
# cost functions
def cost_OLS(X, y, theta, **kwargs):
    return .5*np.mean((y-X @ theta)**2)

def cost_Ridge(X, y, theta, **kwargs):
    lmbda = kwargs['lambda']
    return .5*np.mean((y -  X @ theta )**2) + .5*np.mean(theta**2)

def cost_Lasso(X, y, theta, **kwargs):
    lmbda = kwargs['lambda']
    return .5*np.mean( ( y- X @ theta )**2 ) + .5*np.mean(np.abs(theta))

from autograd import grad
def grad_desc(X, y, cost_func, eta, idx_deriv=2, theta_init = None, check_conv = False, tol=1e-8, store_thetas = False, max_iter=int(1e4), **kwargs):
    # if init vals of theta is not given init theta with random normal distr vals
    if theta_init is None:
        theta = np.random.normal(size=(X.shape[1]))
    # remember to set correct idx_deriv for a cost func not defined with the same order of args as above
    train_grad = grad(cost_func, idx_deriv)
    # store the parameter for analysis
    if store_thetas:
        thetas = np.ones((max_iter, theta.shape[0]))*np.nan
    # init bconv
    bconv = False
    # start GD
    for niter in range(max_iter):
        # compute gradient with autograd
        gradients = train_grad(X, y, theta, **kwargs)
        # update theta by subtracting product of learning rate and gradients
        theta_old = theta.copy()
        theta -= eta*gradients
        # if user wants to check for convergence, when gradient disappears 
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

def grad_desc_mom(X, y, cost_func, eta, delta, idx_deriv=2, theta_init = None, check_conv = False, tol=1e-8, store_thetas = False, max_iter=int(1e4), **kwargs ):
    if theta_init is None:
        theta = np.random.normal(size=(X.shape[1]))
    else:
        theta = theta_init
    # check if delta is set to correct value
    assert delta >= 0 and delta <= 1, ("delta is outside the interval [0,1]")
    # init convergence boolean
    bconv = False
    # setting up gradient, remember correct index for derivative
    train_grad = grad(cost_func, idx_deriv)
    # init moments
    vt = np.zeros(theta.shape)
    # start GD w momentum
    for niter in range(max_iter):
        # compute grad with autograd
        gradients = train_grad(X, y, theta, **kwargs)
        vtmin1 = vt.copy()
        vt = delta*vtmin1 + eta*gradients
        theta_old = theta.copy()
        theta = theta - vt 
        if check_conv:
            # compute difference between curr and prev theta
            del_theta = np.abs(theta - theta_old)
            # check the tolerance condition
            if max(del_theta) <= tol:
                # if converged set conv bool to True
                bconv = True
                return theta, bconv
    return theta, bconv

def divide_batches(X, nmb, batch_pick):
    n = X.shape[0]
    nsamples = n//nmb # number of samples per batch
    inds = np.arange(n)
    # shuffle indices if randomized batches
    if batch_pick == "random":
        np.random.shuffle(inds)# = inds[np.random.randint(n, size=n)]
    # divide in clusters
    if batch_pick == "clusters":
        # do stuff
        return 0
    Binds = [inds[nsamples*i:nsamples*(i+1)] for i in range(nmb)]
    return Binds

def learning_schedule(**kwargs):
    t, t0, t1 = [kwargs[key] for key in ['t', 't0', 't1']]
    if kwargs['scheme'] == "time_decay_rate":
        out =  t0/np.sqrt(t + t1)
    if kwargs['scheme'] == 'exp':
        out = t0* np.exp(-t1*t)
    return out
def SGD(X, y, cost_func, eta, delta, nepochs, nmb, 
        batch_pick="random", idx_deriv=2, theta_init = None, 
        tol=1e-8, store_thetas = False, max_iter=int(1e4), **kwargs ):
    # initialise batches of indices
    Binds = divide_batches(X, nmb, batch_pick)
    # init theta
    if theta_init is None: theta = np.random.normal(size=(X.shape[1]))
    else: theta = theta_init
    bconv = False
    # setting up gradient, remember correct index for derivative
    train_grad = grad(cost_func, idx_deriv)
    # init moments
    vt = np.zeros(theta.shape)
    # init sequence of theta
    theta_seq  = []; theta_seq.append(theta)
    for iepoch in range(nepochs): # iterating over nr of epochs
        for imb in range(nmb): # iterating over mini batches
            count = iepoch*nmb + imb + 1
            ibatch = np.random.randint(nmb)
            inds_k = Binds[ibatch]
            gradients = (1/nmb)*train_grad(X[inds_k], y[inds_k], theta, **kwargs)
            kwargs['t'] = count
            eta_t = learning_schedule(**kwargs)
            vtmin1 = vt.copy()
            vt = delta*vtmin1 + eta_t*gradients
            theta_old = theta.copy()
            theta = theta - vt
            del_theta = np.abs(theta - theta_old)
            # check the tolerance condition
            if max(del_theta) <= tol: bconv = True; return theta, bconv
    return theta, bconv

def ADAgrad_gd(X, y, cost_func, eta, gamma, delta, w_mom, nmb, nepochs, batch_pick='random', idx_deriv=2, theta_init = None,
                check_conv = False, tol=1e-8, store_thetas = False, max_iter=int(1e4), **kwargs):
    # init theta
    if theta_init is None: theta = np.random.normal(size=(X.shape[1]))
    else: theta = theta_init
    bconv = False
    # setting up gradient, remember correct index for derivative
    train_grad = grad(cost_func, idx_deriv)
    # init moments and learning rate
    vt = np.zeros(theta.shape)
    # init acum square gradient
    G = np.zeros((X.shape[1], X.shape[1]))
    for i in range(max_iter):
        count = i+ 1
        gradients = (1/nmb)*train_grad(X, y, theta, **kwargs)
        G += gradients @ gradients.T
        eta_t = (np.c_[eta/(delta + np.sqrt( np.diagonal(G) ))]).ravel()
        if w_mom:
            assert gamma is not None, ("To run with momentum we need gamma parameter")
            vtmin1 = vt.copy()
            vt = gamma*vtmin1 + np.multiply(eta_t,gradients)
        else:
            vt = np.multiply(eta_t,gradients)
        theta_old = theta.copy()
        theta = theta - vt
        if check_conv:
            # compute difference between curr and prev theta
            del_theta = np.abs(theta - theta_old)
            # check the tolerance condition
            if max(del_theta) <= tol:
                # if converged set conv bool to True
                bconv = True
                return theta, bconv
    return theta, bconv
def ADAgrad_sgd(X, y, cost_func, eta, gamma, delta, w_mom, nmb, nepochs, batch_pick='random', idx_deriv=2, theta_init = None,
                check_conv = False, tol=1e-8, store_thetas = False, max_iter=int(1e4), **kwargs):
    # initialise batches of indices
    Binds = divide_batches(X, nmb, batch_pick)
    # init theta
    if theta_init is None: theta = np.random.normal(size=(X.shape[1]))
    else: theta = theta_init
    bconv = False
    # setting up gradient, remember correct index for derivative
    train_grad = grad(cost_func, idx_deriv)
    # init moments
    vt = np.zeros(theta.shape)
    # init acum square gradient
    G = np.zeros((X.shape[1], X.shape[1]))
    for iepoch in range(nepochs): # iterating over nr of epochs
        for imb in range(nmb): # iterating over mini batches
            count = iepoch*nmb + imb + 1
            ibatch = np.random.randint(nmb)
            inds_k = Binds[ibatch]
            gradients = (1/nmb)*train_grad(X[inds_k], y[inds_k], theta, **kwargs)
            G += gradients @ gradients.T
            eta_t = (np.c_[eta/(delta + np.sqrt( np.diagonal(G) ))]).ravel()
            if w_mom:
                assert gamma is not None, ("To run with momentum we need gamma parameter")
                vtmin1 = vt.copy()
                vt = gamma*vtmin1 + np.multiply(eta_t,gradients)
            else:
                vt = np.multiply(eta_t,gradients)
            theta_old = theta.copy()
            theta = theta - vt
            if check_conv:
                # compute difference between curr and prev theta
                del_theta = np.abs(theta - theta_old)
                # check the tolerance condition
                if max(del_theta) <= tol:
                    # if converged set conv bool to True
                    bconv = True
                    return theta, bconv
    return theta, bconv

def RMSprop(X, y, cost_func, eta, delta, beta, w_mom, nmb, nepochs, gamma = None, batch_pick='random', idx_deriv=2, theta_init = None,
            check_conv = False, tol=1e-8, store_thetas = False, max_iter=int(1e4), **kwargs):
    # initialise batches of indices
    Binds = divide_batches(X, nmb, batch_pick)
    # init theta
    if theta_init is None: theta = np.random.normal(size=(X.shape[1]))
    else: theta = theta_init
    bconv = False
    # setting up gradient, remember correct index for derivative
    train_grad = grad(cost_func, idx_deriv)
    # init moments
    eta_t = eta
    vt = np.zeros(theta.shape)
    # init acum square gradient
    G = np.zeros((X.shape[1], X.shape[1]))
    for iepoch in range(nepochs): # iterating over nr of epochs
        for imb in range(nmb): # iterating over mini batches
            count = iepoch*nmb + imb + 1
            ibatch = np.random.randint(nmb)
            inds_k = Binds[ibatch]
            # compute gradients
            gradients = (1/nmb)*train_grad(X[inds_k], y[inds_k], theta, **kwargs)
            # copy old acum outer prod grads
            G_old = G.copy()
            # compute new acum outer prod gradas
            G = beta*G_old + (1-beta)*gradients @ gradients.T
            # update learning parameter
            eta_t = (np.c_[eta/(delta + np.sqrt( np.diagonal(G) ))]).ravel()
            if w_mom:
                assert gamma is not None, ("To run with momentum we need gamma parameter")
                vtmin1 = vt.copy()
                vt = gamma*vtmin1 + np.multiply(eta_t,gradients)
            else:
                vt = np.multiply(eta_t,gradients)
            theta_old = theta.copy()
            theta = theta - vt
            if check_conv:
                # compute difference between curr and prev theta
                del_theta = np.abs(theta - theta_old)
                # check the tolerance condition
                if max(del_theta) <= tol:
                    # if converged set conv bool to True
                    bconv = True
                    return theta, bconv
        count += 1
    return theta, bconv

def ADAM(X, y, cost_func, eta, delta, beta1, beta2, w_mom, nmb, nepochs, batch_pick='random', idx_deriv=2, theta_init = None,
            check_conv = False, tol=1e-8, store_thetas = False, max_iter=int(1e4), **kwargs):
    """
    ADAM by defninition from the paper by Kingma & Ba.
    """
    
    # initialise batches of indices
    Binds = divide_batches(X, nmb, batch_pick)
    # init theta
    if theta_init is None: theta = np.random.normal(size=(X.shape[1]))
    else: theta = theta_init
    bconv = False
    # setting up gradient, remember correct index for derivative
    train_grad = grad(cost_func, idx_deriv)
    # init acum square gradient
    G = np.zeros((X.shape[1], X.shape[1]))
    # init moments
    mt = np.zeros(theta.shape)
    vt = np.zeros(theta.shape)
    # init counts
    for iepoch in range(nepochs): # iterating over nr of epochs
        for imb in range(nmb): # iterating over mini batches
            count = iepoch*nmb + imb + 1
            ibatch = np.random.randint(nmb)
            inds_k = Binds[ibatch]
            gradients = (1/nmb)*train_grad(X[inds_k], y[inds_k], theta, **kwargs)
            # update biased first moment
            mtmin1 = mt.copy()
            mt = beta1*mtmin1 + (1 - beta1)*gradients
            # update biased second moment
            G_old = G.copy()
            G = beta2*G_old + (1-beta2)*gradients @ gradients.T
            # update bias corrected first moment
            mthat = mt/(1 - beta1**count)
            # update bias corrected second moment
            Ghat = G/( 1 - beta2**count)
            # update learning parameter
            eta_t = (np.c_[eta/(delta + np.sqrt( np.diagonal(Ghat) ))]).ravel()
            # update theta
            theta_old = theta.copy()
            theta = theta - eta_t*mthat
            if check_conv and niter > 0:
                # compute difference between curr and prev theta
                del_theta = np.abs(theta - theta_old)
                # check the tolerance condition
                if max(del_theta) <= tol:
                    # if converged set conv bool to True
                    bconv = True
                    return theta, bconv
    return theta, bconv
