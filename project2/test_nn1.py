# test cript for the NN
from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(3155)
x = np.linspace(-1,1,1000)
# using simple linear regression to validate the gradient descent methods
#X = np.ones((x.shape[0],2))
#X[:,1] = x
y = test_func_poly_deg_p(deg = 2, avec = [0,1,2], x = x)
ynoisy = y + np.random.normal(0,.1,size=y.shape)
x_pp = (x - np.mean(x))/np.std(x) # pre-processing of input
nn = Neural_Network(x_pp[:,np.newaxis],  y[:,np.newaxis], costfunc = 'ols')
nn.add_layer(nodes = 5, af = 'tanh')
nn.output_layer(af = 'tanh')
for iter in range(1000):
    nn.feed_forward()
    nn.backpropagation()
    for l in range(1,nn.layers):
        nn.weights[l] -= float(1)*nn.nabla_w_C[l]
        nn.bias[l] -= float(1)*nn.nabla_b_C[l]
    nn.feed_forward()

