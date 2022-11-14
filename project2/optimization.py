# optimization script with all the optimization routines
import numpy as np
import autograd.numpy as np
# cost functions
def cost_OLS(X, y, theta, **kwargs):
    return .5*np.sum((y-X @ theta)**2)/y.shape[0]

def cost_Ridge(X, y, theta, **kwargs):
    lmbda = kwargs['lambda']
    return .5*np.sum((y -  X @ theta )**2)/y.shape[0] + lmbda*np.sum(theta**2)

def cost_Lasso(X, y, theta, **kwargs):
    lmbda = kwargs['lambda']
    return .5*np.sum( ( y- X @ theta )**2 )/y.shape[0] + lmbda*np.sum(np.abs(theta))

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
    def __init__(self, X, y, cost_func, eta, gamma = 1e-3, delta = 1e-8, beta1 = 1e-3, beta2 = 1e-3, idx_deriv=2, theta_init = None, 
                 check_conv = False, tol=1e-8, store_thetas = False, max_iter=int(1e4), batch_pick = None, w_mom = False, **kwargs):
        self.max_iter = max_iter
        self.tol = tol
        self.kwargs = kwargs
        self.w_mom = w_mom
        # set hyperparameters
        self.eta0 = eta
        self.eta = eta
        self.delta = delta
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        # set cost function
        self.cost_func = cost_func
        # set full data
        self.nsamples = X.shape[0]
        self.X_data_full = X
        self.Y_data_full = y
        # set convergence boolean to False
        self.bconv = False
        # init gradient function from autograd
        self.train_grad = grad(cost_func, idx_deriv)
        self.theta = np.random.normal(size=(X.shape[1], 1))
        # init first and second moment respectively
        self.mt = np.zeros(self.theta.shape)
        self.vt = np.zeros(self.theta.shape)
        # init first and second moment from prev iteration respectively
        self.mtmin1 = np.zeros(self.theta.shape)
        self.vtmin1 = np.zeros(self.theta.shape)
        # init the outer product of the gradients
        self.G = np.zeros((X.shape[1], X.shape[1]))
        # if init vals of theta is not given init theta with random normal distr vals
        self.batch_pick = batch_pick

    def __call__(self, method, epochs = None, size_mini_batches = None, 
                 w_mom = False, verbose = False, store_mse = False, **kwargs):
        if method not in ['grad_desc', 'grad_desc_mom', 'ADAgrad_gd']:
            assert epochs is not None, ("Have to assign epochs for stochastic methods")
            assert size_mini_batches is not None, ("Have to assign size of batches for stochastic methods")
            # number of mini batche# number of mini batches
            nmb = self.nsamples//size_mini_batches
            if store_mse:
                self.mse = np.zeros((epochs, nmb))
            for iepoch in range(epochs): # iterating over nr of epochs
                for imb in range(nmb): # iterating over mini batches
                    count = iepoch*nmb + imb + 1
                    theta_old = self.theta.copy()
                    ibatch = np.random.randint(self.nsamples - size_mini_batches)
                    inds_k = divide_batches(self.X_data_full, nmb, istart = ibatch)
                    gradients = (1/size_mini_batches)*self.train_grad(self.X_data_full[inds_k], self.Y_data_full[inds_k], self.theta, **kwargs)
                    if verbose:
                        print("Epoch {0}/{1}, batch {2}/{3}".format(iepoch+1,epochs,imb+1,nmb ))
                        print("Max delta theta: ", np.max(self.theta - theta_old))
                    if method == 'sgd' : 
                        self.SGD(gradients)
                        if kwargs['scheme'] == 'linear':
                            kwargs['t'] = self.eta0
                        else:
                            kwargs['t'] = iepoch*nmb + imb + 1
                        self.eta = self.learning_schedule(**kwargs)
                    
                    if method == 'adagrad_sgd': self.ADAgrad(gradients)
                    if method == 'rms_prop' : self.RMSprop(gradients)
                    if method == 'adam' : self.ADAM(gradients, count)
                    if store_mse: self.mse[iepoch, imb] = cost_OLS(self.X_data_full,self.Y_data_full,  self.theta)
                    #self.bconv = self.check_conv(theta_old)
                    self.bconv = np.sum(np.abs(gradients)) <= self.tol
                    if self.bconv: break
                if self.bconv: break
        else:
            for i in range(self.max_iter):
                theta_old = self.theta.copy()
                gradients = self.train_grad(self.X_data_full, self.Y_data_full, self.theta, **kwargs)
                if method == 'grad_desc': 
                    self.grad_desc(gradients)
                if method == 'grad_desc_mom': 
                    self.grad_desc(gradients)
                if method == 'adagrad':
                    self.ADAgrad(gradients)
                self.bconv = self.check_conv(theta_old)
                if self.bconv: break
    def grad_desc(self, gradients, **kwargs):
        self.theta -= self.eta*gradients
        
    def grad_desc_mom(self, gradients, **kwargs ):
        # check if delta is set to correct value
        assert self.gamma >= 0 and self.gamma <= 1, ("Gamma is outside the interval [0,1]")
        # start GD w momentum
        self.vtmin1 = self.vt.copy()
        self.vt = self.gamma*self.vtmin1 + self.eta*gradients
        self.theta -= self.vt 

    def learning_schedule(self,**kwargs):
        t, t0, t1 = [kwargs[key] for key in ['t', 't0', 't1']]
        if kwargs['scheme'] == "time_decay_rate":
            out =  t0/np.sqrt(t + t1)
        if kwargs['scheme'] == 'exp':
            out = t0* np.exp(-t1*t)
        if kwargs['scheme'] == 'linear':
            out = t
        return out
    def check_conv(self, theta_old):
        if np.max(np.abs(self.theta-theta_old)) <= self.tol: return True
        else: return False

    # set the length of the mini batches to a number which is a power of 2
    def SGD(self, gradients, **kwargs ):
        if self.w_mom:
            self.grad_desc_mom(gradients)
        else:
            self.grad_desc(gradients)
           

    def ADAgrad(self,gradients, **kwargs):
        self.G += gradients @ gradients.T
        eta_t = (np.c_[self.eta/(self.delta + np.sqrt( np.diagonal(self.G) ))])
        if self.w_mom:
            assert self.gamma is not None, ("To run with momentum we need gamma parameter")
            self.vtmin1 = self.vt.copy()
            self.vt = self.gamma*self.vtmin1 + np.multiply(eta_t,gradients)
        else:
            self.vt = np.multiply(eta_t,gradients)
        self.theta -= self.vt
    
    def RMSprop(self,gradients, **kwargs):
        # copy old acum outer prod grads
        self.G_old = self.G.copy()
        # compute new acum outer prod gradas
        self.G = self.beta2*self.G_old + (1-self.beta2)*gradients @ gradients.T
        # update learning parameter
        eta_t = (np.c_[self.eta/(self.delta + np.sqrt( np.diagonal(self.G) ))])
        if self.w_mom:
            assert self.gamma is not None, ("To run with momentum we need gamma parameter")
            self.vtmin1 = self.vt.copy()
            self.vt = self.gamma*self.vtmin1 + np.multiply(eta_t,gradients)
        else:
            self.vt = np.multiply(eta_t,gradients)
        self.theta -= self.vt

    def ADAM(self,gradients,count, **kwargs):
        """
        ADAM by defninition from the paper by Kingma & Ba.
    """
        # update biased first moment
        self.mtmin1 = self.mt.copy()
        self.mt = self.beta1*self.mtmin1 + (1 - self.beta1)*gradients
        # update biased second moment
        self.G_old = self.G.copy()
        self.G = self.beta2*self.G_old + (1-self.beta2)*gradients @ gradients.T
        # update bias corrected first moment
        mthat = self.mt/(1 - self.beta1**count)
        # update bias corrected second moment
        Ghat = self.G/( 1 - self.beta2**count)
        # update learning parameter
        eta_t = (np.c_[self.eta/(self.delta + np.sqrt( np.diagonal(Ghat) ))])
        # update theta
        self.theta -= eta_t*mthat
