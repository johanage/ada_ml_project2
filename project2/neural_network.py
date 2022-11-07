# all the code for the neural network
# column vectors are by definition NOT transpose
# row vectors are transpose such that it follow the theoretical defnition Matrix (n,m) @ Column vector (m,1) = Column vector (n,1)
from autograd import grad
import numpy as np
import autograd.numpy as np
from optimization import divide_batches
# defining the cost functions so that they work with autograd
# and can be optimized wrt the output activations
def cost_ols(y, a, **kwargs):
    return np.mean(np.square(y-a))
def cost_ridge(y, a, w, **kwargs):
    lmbda = kwargs['lambda']
    return np.mean(np.square(y-a)) + lmbda*np.mean(np.square(w))
def cost_lasso(y, a, w, **kwargs):
    lmbda = kwargs['lambda']
    return np.mean(np.square(y-a)) + lmbda*np.mean(np.abs(w))

class Neural_Network(object):
    def __init__(self, X, y, costfunc, eta):
        """
        The initialization of the NN.

        Args:
        X - ndarray, the input data (nsamples, nfeatures)
        y - ndarray, the target data
        costfunc - str, specification of type of cost function 
        eta - float, learning rate
        """
        self.target = y
        self.Xdata_full = X
        self.Ydata_full = y
        self.layers = 0
        self.nodes = {0 : X.shape[1]}
        self.afs = {}
        self.costfunc = costfunc
        self.grad_cost()
        self.a = {0 : X}
        self.weights = {}
        self.bias = {}
        self.z = {}
        # init the derivatives
        self.nabla_w_C = {}
        self.nabla_b_C = {}
        self.eta = eta
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
        if weights is not None: self.weights[l] = weights
        else: self.weights[l] = np.random.normal(0,0.5, size=(self.nodes[l-1], self.nodes[l] ) ) 
        if bias is not None: self.bias[l] = bias
        else: self.bias[l] = np.random.normal(0, 0.5, size=(self.nodes[l],1) )
   
    def output_layer(self, af, weights = None, bias = None):
        """
        Adds output layer to the NN.
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
        else: self.weights[l] = np.random.normal(0,0.5, size=(self.nodes[l-1], self.nodes[l] ) ) 
        if bias is not None: self.bias[l] = bias
        else: self.bias[l] = np.random.normal(0, 0.5, size=(self.nodes[l], 1) ) 
        
    
    def feed_forward(self):
        """
        Feed forward algorithm for all layers.
        l - int, starts at 0, e.g. a[0] = X (input)
        """
        for l in range(1, self.layers+1):
            self.z[l] = self.a[l-1] @ self.weights[l] + self.bias[l].T
            self.a[l] = self.sigma(self.afs[l], self.z[l].copy()) 
        
    def grad_cost(self):
        """
        Computes the gradient of the cost function wrt
        the output activations at layer l
        """
        if self.costfunc == 'ols':            
            self.nabla_a_C = grad(cost_ols, 1)
        if self.costfunc == 'ridge':    
            self.nabla_a_C = grad(cost_ridge, 1)
        if self.costfunc == 'lasso':
            self.nabla_a_C = grad(cost_lasso, 1)
    
    def delta_L(self, **kwargs):
        # compute the gradient for the last layer to start backpropagation
        gradient = self.nabla_a_C(self.target, self.a[self.layers], **kwargs)
        dL = np.multiply(gradient, self.sigma_prime(self.afs[self.layers], self.z[self.layers]) )
        self.delta = {self.layers : dL.copy()}

    def delta_l(self,l, **kwargs):
        #dl = np.multiply(self.weights[l+1].T @ self.delta[l+1], self.sigma_prime(self.afs[l], self.z[l]) )
        dl = np.multiply(self.delta[l+1] @ self.weights[l+1].T, self.sigma_prime(self.afs[l], self.z[l]) )
        self.delta[l] = dl.copy()
        
    def backpropagation(self, **kwargs):
        # compute the error in layer
        self.delta_L(**kwargs)
        for l in range(self.layers):
            idx = self.layers-l
            if l > 0:
                self.delta_l(idx)
            self.nabla_w_C[idx] = self.a[idx-1].T @ self.delta[idx]
            self.nabla_b_C[idx] = np.sum(self.delta[idx], axis = 0, keepdims=True)
    
    def update_weights(self, size_mini_batches):
        for l in range(1, self.layers):
            self.bias[l]    -= self.eta*self.nabla_b_C[l].T/size_mini_batches
            self.weights[l] -= self.eta*self.nabla_w_C[l]/size_mini_batches
    
    
    def SGD(self, epochs, mini_batches, **kwargs):
        binds = divide_batches(X = self.Xdata_full, nmb = mini_batches, batch_pick = 'random')    
        for nepoch in range(epochs):
            for m in range(mini_batches):
                ind_batch = np.random.randint(mini_batches)
                self.a[0] = self.Xdata_full[binds[ind_batch]] # binds[ind_batch] are random indices according to the batch drawn by divide_batches
                self.target = self.Ydata_full[binds[ind_batch]]
                self.feed_forward()
                self.backpropagation()
                self.update_weights(len(binds[ind_batch]))
                loss = cost_ols(self.target, self.a[self.layers], **kwargs)
                #print(loss)
                print("Epoch {0}/{1}, batch {2}/{3}, loss: {4:.4f}".format(nepoch+1,epochs,m+1,mini_batches,np.mean(loss)) )
        # reset the target and the 0th activation output to y and X?
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
            out = np.maximum(zl, 0)
        if af == "leaky_relu":
            out = .5*( (1 - np.sign(zl))*1e-2*zl + (1 + np.sign(zl))*zl)
        return out
    
    def sigma_prime(self, af, zl):
        """
        Args:
        af - string, activation function
        zl - ndarray, neurons weighted input at layer l
        """
        if af == "linear":
            out = 1
        if af == "sigmoid":
            s = self.sigma(af, zl)
            out = s.copy()*(1 - s.copy())
        if af == "tanh":
            out = 1/np.cosh(zl)
        if af == "relu":
            out = .5*(1 + np.sign(zl))
        if af == "leaky_relu":
            out = .5*( (1 - np.sign(zl))*1e-2 + (1 + np.sign(zl)))
        return out
