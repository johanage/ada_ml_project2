# optimization script with all the optimization routines
import numpy as np
import autograd.numpy as np
# cost functions
def cost_OLS(X, y, theta, **kwargs):
    return .5*np.mean((y-X @ theta)**2)

def cost_Ridge(X, y, theta, **kwargs):
    lmbda = kwargs['lambda']
    return .5*np.mean((y -  X @ theta )**2) + lmbda*np.sum(theta**2)

def cost_Lasso(X, y, theta, **kwargs):
    lmbda = kwargs['lambda']
    return .5*np.mean( ( y- X @ theta )**2 ) + lmbda*np.sum(np.abs(theta))

from autograd import grad

def divide_batches(X, nmb, istart):
    n = X.shape[0]
    nsamples = n//nmb # number of samples per batch
    inds = np.arange(n)
    binds = inds[istart:istart+nsamples]
    return binds


"""
Have to update the functions such that it can be reused in the NN by making the gradient functions only take in the previous gradient,
and make a class of optimizers.
"""

class optimizers:
    def __init__(X, y, cost_func, eta, gamma = 1e-3, delta = 1e-8, beta1 = 1e-3, beta2 = 1e-3, idx_deriv=2, theta_init = None, 
                 check_conv = False, tol=1e-8, store_thetas = False, max_iter=int(1e4), batch_pick = None, **kwargs):
        self.max_iter = max_iter
        self.tol = tol
        self.kwargs = kwargs
        # set hyperparameters
        self.eta = eta
        self.delta = delta
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.cost_func = cost_func
        self.X_data_full = X
        self.Y_data_full = y
        self.bonv = False
        self.train_grad = grad(cost_func, idx_deriv)
        self.theta = theta_init
        self.mt = np.zeros(theta.shape)
        self.vt = np.zeros(theta.shape)
        self.mtmin1 = np.zeros(theta.shape)
        self.vtmin1 = np.zeros(theta.shape)
        self.G = np.zeros((X.shape[1], X.shape[1]))
        # if init vals of theta is not given init theta with random normal distr vals
        if theta_init is None:
            self.theta = np.random.normal(size=(X.shape[1]))
        self.batch_pick = batch_pick
        #for niter in range(max_iter):
        
    def grad_desc(self, gradients, **kwargs):
        self.update_theta = self.theta - self.eta*gradients

    def grad_desc_mom(self, gradients, **kwargs ):
        # check if delta is set to correct value
        assert delta >= 0 and delta <= 1, ("delta is outside the interval [0,1]")
        # start GD w momentum
        self.vtmin1 = self.vt.copy()
        self.vt = self.gamma*vtmin1 + self.eta*gradients
        self.theta = self.theta - self.vt 

    def learning_schedule(self,**kwargs):
        t, t0, t1 = [kwargs[key] for key in ['t', 't0', 't1']]
        if kwargs['scheme'] == "time_decay_rate":
            out =  t0/np.sqrt(t + t1)
        if kwargs['scheme'] == 'exp':
            out = t0* np.exp(-t1*t)
        return out

    # set the length of the mini batches to a number which is a power of 2
    def SGD(self, size_mini_batches, **kwargs ):
        nmb = size_mini_batches
        # initialise batches of indices
        # init sequence of theta
        for iepoch in range(nepochs): # iterating over nr of epochs
            for imb in range(nmb): # iterating over mini batches
                kwargs['t'] = iepoch*nmb + imb + 1
                ibatch = np.random.randint(nmb)
                inds_k = divide_batches(self.X_data_full, nmb, istart = ibatch)
                gradients = (1/nmb)*train_grad(self.X_data_full[inds_k], self.Y_data_full[inds_k], self.theta, **kwargs)
                eta_t = learning_schedule(**kwargs)
                self.vtmin1 = vt.copy()
                self.vt = self.gamma*vtmin1 + eta_t*gradien>ts
                self.theta = self.theta - self.vt

    def ADAgrad_gd(gradients, **kwargs):
            self.G += gradients @ gradients.T
            eta_t = (np.c_[self.eta/(self.delta + np.sqrt( np.diagonal(self.G) ))]).ravel()
            if w_mom:
                assert self.gamma is not None, ("To run with momentum we need gamma parameter")
                self.vtmin1 = vt.copy()
                self.vt = self.gamma*self.vtmin1 + np.multiply(eta_t,gradients)
            else:
                self.vt = np.multiply(eta_t,gradients)
            self.theta = self.theta - self.vt
    
    def ADAgrad_sgd(self, size_mini_batches, **kwargs):
        nmb = self.Y_data_full.shape[0]//size_mini_batches
        # initialise batches of indices
        for iepoch in range(nepochs): # iterating over nr of epochs
            for imb in range(nmb): # iterating over mini batches
                count = iepoch*nmb + imb + 1
                ibatch = np.random.randint(nmb)
                inds_k = divide_batches(self.X_data_full, nmb, istart = ibatch)
                gradients = (1/nmb)*train_grad(self.X_data_full[inds_k], self.Y_data_full[inds_k], self.theta, **kwargs)
                self.G += gradients @ gradients.T
                eta_t = (np.c_[eta/(delta + np.sqrt( np.diagonal(G) ))]).ravel()
                if w_mom:
                    assert self.gamma is not None, ("To run with momentum we need gamma parameter")
                    self.vtmin1 = vt.copy()
                    self.vt = gamma*vtmin1 + np.multiply(eta_t,gradients)
                else:
                    self.vt = np.multiply(eta_t,gradients)
                self.theta = self.theta - self.vt

    def RMSprop(self,gradients,size_mini_batches, **kwargs):
        nmb = self.Y_data_full.shape[0]//size_mini_batches
        for iepoch in range(nepochs): # iterating over nr of epochs
            for imb in range(nmb): # iterating over mini batches
                ibatch = np.random.randint(nmb)
                inds_k = divide_batches(self.X_data_full, nmb, istart = ibatch)
                gradients = (1/nmb)*train_grad(self.X_data_full[inds_k], self.Y_data_full[inds_k], self.theta, **kwargs)
                # copy old acum outer prod grads
                self.G_old = G.copy()
                # compute new acum outer prod gradas
                G = self.beta2*self.G_old + (1-self.beta2)*self.gradients @ self.gradients.T
                # update learning parameter
                eta_t = (np.c_[self.eta/(self.delta + np.sqrt( np.diagonal(self.G) ))]).ravel()
                if w_mom:
                    assert self.gamma is not None, ("To run with momentum we need gamma parameter")
                    self.vtmin1 = self.vt.copy()
                    self.vt = self.gamma*self.vtmin1 + np.multiply(eta_t,self.gradients)
                else:
                    self.vt = np.multiply(eta_t,self.gradients)
                self.theta = self.theta - self.vt

    def ADAM(self,size_mini_batches, **kwargs):
        """
        ADAM by defninition from the paper by Kingma & Ba.
        """
        nmb = self.Y_data_full.shape[0]//size_mini_batches
        # init counts
        for iepoch in range(nepochs): # iterating over nr of epochs
            for imb in range(nmb): # iterating over mini batches
                count = iepoch*nmb + imb + 1
                ibatch = np.random.randint(nmb)
                inds_k = divide_batches(self.X_data_full, nmb, istart = ibatch)
                gradients = (1/nmb)*train_grad(self.X_data_full[inds_k], self.Y_data_full[inds_k], self.theta, **kwargs)
                # update biased first moment
                self.mtmin1 = mt.copy()
                self.mt = self.beta1*self.mtmin1 + (1 - self.beta1)*self.gradients
                # update biased second moment
                self.G_old = self.G.copy()
                self.G = self.beta2*self.G_old + (1-self.beta2)*self.gradients @ self.gradients.T
                # update bias corrected first moment
                mthat = self.mt/(1 - self.beta1**count)
                # update bias corrected second moment
                Ghat = self.G/( 1 - self.beta2**count)
                # update learning parameter
                eta_t = (np.c_[self.eta/(self.delta + np.sqrt( np.diagonal(self.Ghat) ))]).ravel()
                # update theta
                self.theta = self.theta - eta_t*self.mthat
