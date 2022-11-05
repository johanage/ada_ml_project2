# all the code for the neural network
from autograd import grad
import numpy as np
import autograd.numpy as np
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
    def __init__(self, X, y, costfunc,
            af = 'sigmoid'):
        """
        The initialization of the NN.

        Args:
        X - ndarray, the input data (nsamples, nfeatures)
        y - ndarray, the target data
        costfunc - str, specification of type of cost function 
        af - str, specification of activation function for that layer
        """
        self.target = y
        self.layers = 1
        self.nodes = [X.shape[0]]
        self.afs = [af]
        self.costfunc = costfunc
        self.a = [X]
        self.weights = []
        self.bias = []
        self.z = []
        
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
        l = self.layers-1
        self.nodes.append(nodes)
        self.afs.append(af) 
        if weights is not None: self.weights.append(weights)
        else: self.weights.append(np.random.normal(0,0.5, size=(nodes, self.nodes[l-1]) ) )
        if bias is not None: self.bias.append(bias)
        else: self.bias.append(np.random.normal(0, 0.5, size=(nodes,1) ) )
   
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
        l = self.layers-1
        self.nodes.append(self.nodes[0])
        self.afs.append(af)
        if weights is not None: self.weights.append(weights)
        else: self.weights.append(np.random.normal(0,0.5, size=(self.nodes[-1], self.nodes[-2]) ) )
        if bias is not None: self.bias.append(bias)
        else: self.bias.append(np.random.normal(0, 0.5, size=(self.nodes[-1],1) ) )
        
    
    def feed_forward(self):
        """
        Feed forward algorithm for all layers.
        l - int, starts at 0, e.g. a[0] = X (input)
        """
        l = 1
        for w, b in zip(self.weights, self.bias):
            z = w @ self.a[l-1] + b
            self.z.append(z)
            self.a.append(self.sigma(self.afs[l], z) )
            l += 1
        
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
        # set up the autodiff of the cost function specified in self.costfunc
        self.grad_cost()
        # compute the gradient for the last layer to start backpropagation
        gradient = self.nabla_a_C(self.target, self.a[-1], **kwargs)
        dL = np.multiply(gradient, self.sigma_prime(self.afs[-1], self.z[-1]) )
        self.delta = [dL]

    def delta_l(self,l, **kwargs):
        dl = np.multiply(self.weights[l+1].T @ self.delta[l+1], self.sigma_prime(self.afs[l], self.z[l]) )
        self.delta.insert(0,dl)
        
    def backpropagation(self, **kwargs):
        # init the derivatives
        self.nabla_w_C = [np.zeros(w.shape) for w in self.weights]
        self.nabla_b_C = [np.zeros(b.shape) for b in self.bias]
        # compute the error in layer
        self.delta_L(**kwargs)
        for l in range(1,self.layers):
            if l > 1:
                self.delta_l(-l)
            self.nabla_w_C[-l] = self.delta[-l] @ self.a[-l-1].T
            self.nabla_b_C[-l] = self.delta[-l]
    
    #def update_minibatch(self):


    def sigma(self, af, zl):
        """
        Args:
        af - string, activation function
        zl - ndarray, neurons weighted input at layer l
        """

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

        if af == "sigmoid":
            s = self.sigma(af, zl)
            out = s*(1 - s)
        if af == "tanh":
            s = self.sigma(af, zl)
            out = 1 - np.square(s)
        if af == "relu":
            out = .5*(1 + np.sign(zl))
        if af == "leaky_relu":
            out = .5*( (1 - np.sign(zl))*1e-2 + (1 + np.sign(zl)))
    
        return out
