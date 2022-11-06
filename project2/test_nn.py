# test cript for the NN
from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
import sys
print(sys.argv)
np.random.seed(3155)
x = np.linspace(-1,1,1000)
# using simple linear regression to validate the gradient descent methods
#X = np.ones((x.shape[0],2))
#X[:,1] = x
y = test_func_poly_deg_p(deg = 2, avec = [0,1,2], x = x)
ynoisy = y + np.random.normal(0,.1,size=y.shape)
#x = (x - np.mean(x))/np.std(x) # pre-processing of input
nn = Neural_Network(x[:,np.newaxis],  y[:,np.newaxis], costfunc = sys.argv[1])
nn.add_layer(nodes = int(sys.argv[2]), af = sys.argv[3])
nn.output_layer(af = sys.argv[4])
for iter in range(1000):
    nn.feed_forward()
    nn.backpropagation()
    for l in range(1, nn.layers):
        nn.weights[l] -= float(sys.argv[5])*nn.nabla_w_C[l]
        nn.bias[l] -= float(sys.argv[5])*nn.nabla_b_C[l]
    nn.feed_forward()

plt.figure()
plt.plot(x, ynoisy, label='ynoisy', ls='--', alpha = 0.5)
plt.plot(x, y, label="y")
plt.plot(x, nn.a[nn.layers], label='NN pred')
plt.legend()
plt.show()
