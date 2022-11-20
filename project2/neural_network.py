# all the code for the neural network
# column vectors are by definition NOT transpose
# row vectors are transpose such that it follow the theoretical defnition Matrix (n,m) @ Column vector (m,1) = Column vector (n,1)
from autograd import grad
import numpy as np
import autograd.numpy as np
from optimization import divide_batches
import matplotlib.pyplot as plt
from optimization import *
from project1 import R2score
from sklearn.model_selection import train_test_split

# defining the cost functions so that they work with autograd
# and can be optimized wrt the output activations
def grad_ols(y, a, **kwargs):
    return (a-y)/y.shape[0]

def grad_ridge(y, a,w, **kwargs):
    return (a-y)/y.shape[0] + kwargs['lambda']*np.sum(w,axis=0).T

def grad_lasso(y,a,w,**kwargs):
    return (a-y)/y.shape[0] + kwargs['lambda']*np.sum(np.sign(w, axis=0)).T

def grad_cross_entropy(y, a, **kwargs):
    return (a - y)/(y.shape[0]*a*(1 - a) ) 

def grad_cross_entropy_l2reg(y, a, w, **kwargs):
    return (a - y)/( y.shape[0]*a*(1 - a) ) + kwargs['lambda']*np.sum(w,axis=0).T

def grad_cross_entropy_l1reg(y, a, w, **kwargs):
    return (a - y)/( y.shape[0]*a*(1 - a) ) + kwargs['lambda']*np.sum(np.sign(w, axis=0)).T

def cost_ols(y, a, **kwargs):
    return .5*np.sum(np.square(y-a))/y.shape[0]

def cost_ridge(y, a, w, **kwargs):
    lmbda = kwargs['lambda']
    return .5*np.sum(np.square(y-a))/y.shape[0] + lmbda*np.sum(np.square(w))

def cost_lasso(y, a, w, **kwargs):
    lmbda = kwargs['lambda']
    return .5*np.sum(np.square(y-a))/y.shape[0]+ lmbda*np.sum(np.abs(w))

def cross_entropy(y, a, **kwargs):
    # add epsilon in log to avoid divide by zero
    epsilon = 1e-10
    return - np.sum( y*np.log(a + epsilon) + (1-y)*np.log(1-a+epsilon) )/y.shape[0]

def cross_entropy_l2reg(y, a, w, **kwargs):
    lmbda = kwargs['lambda']
    # add epsilon in log to avoid divide by zero
    epsilon = 1e-10
    return - np.sum( y*np.log(a+epsilon) + (1-y)*np.log(1-a+epsilon) )/y.shape[0] + .5* lmbda*np.sum(np.square(w))

def cross_entropy_l1reg(y, a, w, **kwargs):
    lmbda = kwargs['lambda']
    # add epsilon in log to avoid divide by zero
    epsilon = 1e-10
    return - np.sum( y*np.log(a+epsilon) + (1-y)*np.log(1-a+epsilon) )/y.shape[0] + lmbda*np.sum(np.abs(w))

# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    return onehot_vector

def probs_to_binary(probabilities):
    return (np.round(probabilities)).astype(int)

def accuracy(y, a):
    return sum( [tuple(y[i]) == tuple(a[i]) for i in range(y.shape[0])] )/y.shape[0]


class Neural_Network(object):
    def __init__(self, X, y, costfunc, eta, symbolic_differentiation = False, devsize = 0.1, rs = 3155,
                 method = 'sgd', w_mom = False, beta1 = 0.9, beta2 = 0.99, gamma = 0.9, delta = 1e-8):
        """
        The initialization of the NN.

        Args:
        X - ndarray, the input data (nsamples, nfeatures)
        y - ndarray, the target data
        costfunc - str, specification of type of cost function 
        eta - float, learning rate
        specify_grad - bool, compute the gradient of the cost function from symbolic differentition
        """
        X, X_dev, y, y_dev = train_test_split(X, y, test_size = devsize, random_state = rs)
        self.target = y
        self.Xdata_full = X
        self.Ydata_full = y
        self.Xdata_dev = X_dev
        self.Ydata_dev = y_dev
        self.layers = 0
        self.nodes = {0 : self.Xdata_full.shape[1]}
        self.afs = {}
        # set costfunc
        self.costfunc = costfunc
        self.cost()
        if symbolic_differentiation:
            self.grad_cost_specific()
        else:
            self.grad_cost()
        self.a = {0 : X}
        self.weights = {}
        self.bias = {}
        self.z = {}
        # init the derivatives
        self.nabla_w_C = {}
        self.nabla_b_C = {}
        self.eta = eta
        # init optimizer
        self.method = method
        self.w_mom = w_mom
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.delta_param = delta
        # init v and moments for weights
        self.vt_w = {}
        self.vtmin1_w = {}
        self.mt_w = {}
        self.mtmin1_w = {}
        self.G_w = {}
        self.Gtmin1_w = {}
        # init v and moments for bias
        self.vt_b = {}
        self.vtmin1_b = {}
        self.mt_b = {}
        self.mtmin1_b = {}
        self.G_b = {}
        self.Gtmin1_b = {}

    def info(self):
        print("nodes ", [(k,v) for k,v in self.nodes.items()])
        print("activation functions ", [(k,v) for k,v in self.afs.items()])
        print("activation output key value.shape ", [(k,v.shape) for k,v in self.a.items()])
        print("weights key value.shape ", [(k,v.shape) for k,v in self.weights.items()])
        print("bias key value.shape ", [(k,v.shape) for k,v in self.bias.items()])
        print("z key value.shape ", [(k,v.shape) for k,v in self.z.items()])
        print("delta key value.shape ", [(k,v.shape) for k,v in self.delta.items()])
        print("nabla_b_C key value.shape ", [(k,v.shape) for k,v in self.nabla_b_C.items()])
        print("nabla_w_C key value.shape ", [(k,v.shape) for k,v in self.nabla_w_C.items()])
    
    def add_layer(self, nodes, af, weights = None, bias = None):
        """
        To add a layer to the NN.
        Args:
        nodes - int, number of nodes in the new layer
        af - str, the activation function type
        weights - ndarray float, init weights; if not given it will be init with ~N(0,.5) 
        bias - ndarray float, init bias; if not given it will be init with ~N(0,.5)
        """
        self.layers += 1
        l = self.layers
        self.nodes[l] = nodes
        self.afs[l] = af 
        self.scale = 1
        #if af == 'sigmoid':
        #    self.scale = np.sqrt(2/(self.a[0].shape[0] + self.target.shape[0]) )
        #else:
        #    self.scale = np.sqrt(6/(self.a[0].shape[0] + self.target.shape[0]) )
        print("Variance of init weihgts: ", self.scale)
        if weights is not None: self.weights[l] = weights
        else: self.weights[l] = np.random.normal(0,self.scale, size=(self.nodes[l-1], self.nodes[l] ) )
        if bias is not None: self.bias[l] = bias
        else: self.bias[l] = np.zeros((self.nodes[l],1) ) + 0.1
        # init weights and bias moments
        self.mt_w[l] = np.zeros(self.weights[l].shape)
        self.mt_b[l] = np.zeros(self.bias[l].shape)
        self.vt_w[l] = np.zeros(self.weights[l].shape)
        self.vt_b[l] = np.zeros(self.bias[l].shape)
        self.G_w[l] = np.zeros((self.weights[l].shape[0], self.weights[l].shape[0]))
        self.G_b[l] = np.zeros((self.bias[l].shape[1],self.bias[l].shape[1]))

    def output_layer(self, af, weights = None, bias = None):
        """
        Adds output layer to the NN for regression type problems where the output shape is equal to the input shape.
        Args:
        nodes - int, number of nodes in the new layer
        af - str, the activation function type
        weights - ndarray float, init weights; if not given it will be init with ~N(0,.5) 
        bias - ndarray float, init bias; if not given it will be init with ~N(0,.5)
        """
        self.layers += 1
        l = self.layers
        self.nodes[l] = self.nodes[0]
        self.afs[l] = af
        if weights is not None: self.weights[l] = weights
        else: self.weights[l] = np.random.normal(0,self.scale, size=(self.nodes[l-1], self.nodes[l] ) )
        if bias is not None: self.bias[l] = bias
        else: self.bias[l] = np.zeros( (self.nodes[l], 1) ) + 0.1
        # init weights and bias moments
        self.mt_w[l] = np.zeros(self.weights[l].shape)
        self.mt_b[l] = np.zeros(self.bias[l].shape)
        self.vt_w[l] = np.zeros(self.weights[l].shape)
        self.vt_b[l] = np.zeros(self.bias[l].shape)
        self.G_w[l] = np.zeros((self.weights[l].shape[0], self.weights[l].shape[0]))
        self.G_b[l] = np.zeros((self.bias[l].shape[1],self.bias[l].shape[1]))

    def cost(self):
        """
        Computes the gradient of the cost function wrt
        the output activations at layer l
        """
        if self.costfunc == 'ols':
            self.C = cost_ols
        if self.costfunc == 'ridge':
            self.C = cost_ridge
        if self.costfunc == 'lasso':
            self.C = cost_lasso
        if self.costfunc == 'cross_entropy':
            self.C = cross_entropy
        if self.costfunc == 'cross_entropy_l1reg':
            self.C = cross_entropy_l1reg
        if self.costfunc == 'cross_entropy_l2reg':
            self.C = cross_entropy_l2reg
    
    def grad_cost_specific(self):
        if self.costfunc == "ols":
            self.nabla_a_C = grad_ols
        if self.costfunc == 'ridge':
            self.nabla_a_C = grad_ridge
        if self.costfunc == 'lasso':
            self.nabla_a_C = grad_lasso
        if self.costfunc == "cross_entropy":
            self.nabla_a_C = grad_cross_entropy
        if self.costfunc == "cross_entropy_l2reg":
            self.nabla_a_C = grad_cross_entropy_l2reg
        if self.costfunc == "cross_entropy_l1reg":
            self.nabla_a_C = grad_cross_entropy_l1reg

    def grad_cost(self):
        """
        Computes the gradient of the cost function using autograd
        wrt the output activations at layer l
        """
        if self.costfunc == 'ols':            
            self.nabla_a_C = grad(cost_ols, 1)
        if self.costfunc == 'ridge':    
            self.nabla_a_C = grad(cost_ridge, 1)
        if self.costfunc == 'lasso':
            self.nabla_a_C = grad(cost_lasso, 1)
        if self.costfunc == 'cross_entropy':
            self.nabla_a_C = grad(cross_entropy, 1)
        if self.costfunc == 'cross_entropy_l1reg':
            self.nabla_a_C = grad(cross_entropy_l1reg, 1)
        if self.costfunc == 'cross_entropy_l2reg':
            self.nabla_a_C = grad(cross_entropy_l2reg, 1)
    
    def feed_forward(self):
        """
        Feed forward algorithm for all layers.
        """
        for l in range(1, self.layers+1):
            self.z[l] = self.a[l-1] @ self.weights[l] + self.bias[l].T
            self.a[l] = self.sigma(self.afs[l], self.z[l].copy()) 
    
    def predict(self, X):
        self.a[0] =  X
        self.feed_forward()
        return self.a[self.layers]

    def delta_L(self, **kwargs):
        # compute the gradient for the last layer to start backpropagation
        if self.costfunc == 'cross_entropy':
            dL = grad_ols(self.target, self.a[self.layers], **kwargs)
        else:
            if self.costfunc == 'ols':
                gradient = self.nabla_a_C(self.target, self.a[self.layers], **kwargs)
            else:
                gradient = self.nabla_a_C(self.target, self.a[self.layers], self.weights[self.layers], **kwargs)
            print("max gradient ", np.max(gradient))
            #self.optimizer_bias = optimizer(X = X, y = y, cost_func = self.C, eta = self.eta, gamma = self.gamma, 
            #                                beta1 = self.beta1, beta2 = self.beta2, w_mom = self.w_mom, theta_init = self.weights)
            sigma_prime_L = self.sigma_prime(self.afs[self.layers], self.z[self.layers])
            dL = np.multiply(gradient, sigma_prime_L )
        self.delta = {self.layers : dL }

    def delta_l(self,l):
        dl = np.multiply(self.delta[l+1] @ self.weights[l+1].T, self.sigma_prime(self.afs[l], self.z[l]) )
        self.delta[l] = dl.copy()
        
    def backpropagation(self, **kwargs):
        # compute the error in layer
        self.delta_L(**kwargs)
        for l in range(self.layers):
            idx = self.layers-l
            if l > 0:
                self.delta_l(idx)
            self.nabla_b_C[idx] = np.sum(self.delta[idx], axis = 0, keepdims=True)
            self.nabla_w_C[idx] = self.a[idx-1].T @ self.delta[idx]
    
    def update_weights(self, size_mini_batches, count = None):
        for l in range(1,self.layers+1):
            if self.method == 'sgd':
                if self.w_mom:
                    self.vtmin1_w[l] = self.vt_w[l].copy()
                    self.vt_w[l] = self.gamma*self.vtmin1_w[l].copy() + self.eta*self.nabla_w_C[l].copy()
                    self.vtmin1_b[l] = self.vt_b[l].copy()
                    self.vt_b[l] = self.gamma*self.vtmin1_b[l].copy() + self.eta*self.nabla_b_C[l].copy().T
                else:
                    self.vt_w[l] = self.eta*self.nabla_w_C[l].copy()
                    self.vt_b[l] = self.eta*self.nabla_b_C[l].copy().T
            # rms prop and adam
            if self.method in ['rms_prop', 'adam']:
                self.Gtmin1_w[l] = self.G_w[l].copy()
                self.Gtmin1_b[l] = self.G_b[l].copy()
                self.G_w[l] = self.beta2*self.Gtmin1_w[l].copy() + (1-self.beta2)*self.nabla_w_C[l].copy() @ self.nabla_w_C[l].copy().T
                self.G_b[l] = self.beta2*self.Gtmin1_b[l].copy() + (1-self.beta2)*self.nabla_b_C[l].copy() @ self.nabla_b_C[l].copy().T
                if self.method == 'rms_prop':
                    eta_t_w = np.c_[self.eta/(self.delta_param + np.sqrt( np.diagonal(self.G_w[l].copy() ) ))]
                    eta_t_b = np.c_[self.eta/(self.delta_param + np.sqrt( np.diagonal(self.G_b[l].copy() ) ))]
                    # with or without momentum
                    if self.w_mom:
                        self.vtmin1_w[l] = self.vt_w[l].copy()
                        self.vtmin1_b[l] = self.vt_b[l].copy()
                        self.vt_w[l] = self.gamma*self.vtmin1_w[l].copy() + np.multiply(eta_t_w,self.nabla_w_C[l].copy() )
                        self.vt_b[l] = self.gamma*self.vtmin1_b[l].copy() + np.multiply(eta_t_b,self.nabla_b_C[l].copy() )
                    else:
                        self.vt_w[l] = np.multiply(eta_t_w,self.nabla_w_C[l].copy() )
                        self.vt_b[l] = np.multiply(eta_t_b,self.nabla_b_C[l].copy() )
                # specific for adam
                if self.method == 'adam':
                    self.mtmin1_w[l] = self.mt_w[l].copy()
                    self.mt_w[l] = self.beta1*self.mtmin1_w[l].copy() + (1 - self.beta1)*self.nabla_w_C[l].copy()
                    self.mtmin1_b[l] = self.mt_b[l].copy()
                    self.mt_b[l] = self.beta1*self.mtmin1_b[l].copy() + (1 - self.beta1)*self.nabla_b_C[l].copy()
                    # update first bias corrected momentum
                    mthat_w = self.mt_w[l].copy()/(1 - self.beta1**count)
                    mthat_b = self.mt_b[l].copy()/(1 - self.beta1**count)
                    # update acumulated outer prod of gradients
                    Ghat_w  = self.G_w[l].copy() /(1 - self.beta2**count)
                    Ghat_b  = self.G_b[l].copy() /(1 - self.beta2**count)
                    # update adaptive learning rate
                    eta_t_w = np.c_[self.eta/(self.delta_param + np.sqrt( np.diagonal(Ghat_w.copy() ) ))]
                    eta_t_b = np.c_[self.eta/(self.delta_param + np.sqrt( np.diagonal(Ghat_b.copy() ) ))]
                    # update second bias corrected momentum 
                    self.vt_w[l] = eta_t_w.copy() * mthat_w.copy()
                    self.vt_b[l] = eta_t_b.copy() * mthat_b.copy()
                    del mthat_w
                    del mthat_b
                    del Ghat_w
                    del Ghat_b
                del eta_t_w
                del eta_t_b
            # update
            #print(self.vt_b[l].copy().shape)
            #print(self.bias[l].shape)
            #assert False
            self.bias[l]    -= np.sum(self.vt_b[l].copy(), axis = 1, keepdims=True)
            self.weights[l] -= self.vt_w[l].copy()


    def SGD(self, epochs, size_mini_batches,tol=1e-4, printout = False, plot=False, 
            store_grads = False, store_activation_output = False, batchnorm = False, **kwargs):
        """
        Stochastic gradient descent for optimizin the weights and biases in the NN.
        Tip: choose number of minibatches s.t. the lenth of each batch is a power of 2.
        
        Args:
        - epochs            - int, number of epochs
        - size_mini_batches - int, number of samples in each mini batch (approx)
        - printout          - bool, prints out information on loss wrt to epoch number and batch number
        - plot              - bool, plots the (epoch number, loss)
        """
        nsamples = self.Ydata_full.shape[0]
        mini_batches = nsamples//size_mini_batches
        print(" # mini batches", mini_batches)
        self.r2 = np.zeros(epochs)
        self.mse = np.zeros(epochs)
        self.losses = np.zeros(epochs)
        self.losses_dev = np.zeros(epochs)
        self.l2norm_gradC_weights = np.ones(epochs)*np.nan
        # storing norm of gradients for analysing convergence
        if store_grads:
            self.grads = {}
            self.grads[0] = self.nabla_w_C.copy()
        # storing activation output for each layer at each iteration
        if store_activation_output:
            self.activation_output = {}
            self.activation_output[0] = self.a.copy()
        for nepoch in range(epochs):
            bias_old = self.bias[self.layers].copy()
            weights_old = self.weights[self.layers].copy()
            for m in range(mini_batches):
                count = nepoch*mini_batches + m + 1
                ind_batch = np.random.randint(nsamples - size_mini_batches)
                binds = divide_batches(X = self.Xdata_full, nmb = mini_batches, istart = ind_batch)
                # bactch normalization
                if batchnorm:
                    data = (self.Xdata_full[binds].copy() - np.sum(self.Xdata_full[binds].copy(), axis = 0)/len(binds))/np.std(self.Xdata_full[binds].copy(), axis = 0)
                else: 
                    data = self.Xdata_full[binds].copy()
                # set the first activation output and target to the batch samples 
                self.a[0] = data.copy() 
                self.target = self.Ydata_full[binds].copy()
                # do feed forward on the batch
                self.feed_forward()
                # do backprop on the batch
                self.backpropagation(**kwargs)
                # update weights and biases
                self.update_weights(size_mini_batches, count)
                # store weights and activation output for anlaysis
                if store_grads:
                    self.grads[count] = self.nabla_w_C.copy()
                if store_activation_output:
                    self.activation_output[count] = self.a.copy()
                # print information
                if printout:
                    print("max of activation output: ", np.max(self.a[self.layers-1]))
                    print("Epoch {0}/{1}, batch {2}/{3}, loss: 1e{4:.4f}".format(nepoch+1,epochs,m+1,mini_batches, np.log10(np.mean(self.losses))) )
                    print("Max delta weights: ", np.max(self.weights[self.layers] - weights_old))
                    print("Max delta bias: ", np.max(self.bias[self.layers] - bias_old))
            # compute loss to indicate performance wrt epoch number
            if self.costfunc in ['ols','cross_entropy']:
                loss = self.C(self.Ydata_full, self.predict(self.Xdata_full), **kwargs)
                loss_dev = self.C(self.Ydata_dev, self.predict(self.Xdata_dev), **kwargs)
            else:
                loss = self.C(self.Ydata_full, self.predict(self.Xdata_full), self.weights[self.layers], **kwargs)
                loss_dev = self.C(self.Ydata_dev, self.predict(self.Xdata_dev), self.weights[self.layers], **kwargs)
            self.losses[nepoch] = loss
            self.losses_dev[nepoch] = loss_dev
            self.r2[nepoch] = 1 - np.sum(np.sum((self.Ydata_full - self.predict(self.Xdata_full))**2, axis=0) / np.sum( (self.Ydata_full - np.sum(self.Ydata_full, axis=0)/self.target[0])**2 ) )#R2score(self.target, self.a[self.layers])
            self.mse[nepoch] = cost_ols(self.Ydata_full, self.predict(self.Xdata_full), **kwargs)
            self.l2norm_gradC_weights[nepoch] = np.sqrt(np.sum((self.nabla_w_C[self.layers])**2))
            if nepoch > 0 and self.l2norm_gradC_weights[nepoch] <=tol: self.bconv = True; break 
            if printout:
                print("Epoch {0}/{1}, l2-norm grad C_weights: 1e{2:.4f}".format(nepoch+1,epochs, np.log10(np.mean(self.l2norm_gradC_weights))) )
        if plot:
            plt.plot(np.arange(epochs) + 1, self.losses)
            plt.xlabel("Epochs")
            plt.ylabel("$C(\\theta)$")
            plt.show()

    def sigma(self, af, zl):
        """
        Args:
        af - string, activation function
        zl - ndarray, neurons weighted input at layer l
        """
        if af == "linear":
            out = zl
        if af == "sigmoid":
            out = 1/(1 + np.exp(-zl))
        if af == "tanh":
            out = np.tanh(zl)
        if af == "relu":
            #out = np.maximum(zl, 1e-8)
            out = np.maximum(zl, np.zeros(zl.shape))
        if af == "leaky_relu":
            out = .5*( (1 - np.sign(zl))*1e-2*zl + (1 + np.sign(zl))*zl)
        if af == "softmax":
            out = np.exp(zl)/np.sum( np.exp(zl), axis=1, keepdims=True)
        
        return out
           
    def sigma_prime(self, af, zl, delta = None):
        """
        Args:
        af - string, activation function
        zl - ndarray, neurons weighted input at layer l
        """
        if af == "linear":
            out = 1
        if af == "sigmoid":
            s = self.sigma(af, zl)
            out = s*(1 - s)
        if af == "tanh":
            out = 1/np.cosh(zl)
        if af == "relu":
            out = .5*(1 + np.sign(zl))
        if af == "leaky_relu":
            out = .5*( (1 - np.sign(zl))*1e-2 + (1 + np.sign(zl)))
        if af == "softmax":
            s = self.sigma(af, zl)
            out = s*(1 - s)
        return out
