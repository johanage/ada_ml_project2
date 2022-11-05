# test cript for the NN
from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
import sys
print(sys.argv)
np.random.seed(3155)
x = np.linspace(-1,1,100)
# using simple linear regression to validate the gradient descent methods
#X = np.ones((x.shape[0],2))
#X[:,1] = x
y = test_func_poly_deg_p(deg = 2, avec = [0,1,2], x = x)
ynoisy = y + np.random.normal(size=y.shape)

nn = Neural_Network(x[:,np.newaxis],  y[:,np.newaxis], costfunc = sys.argv[1])
nn.add_layer(nodes = int(sys.argv[2]), af = sys.argv[3])
nn.output_layer()
for iter in range(500):
    nn.feed_forward()
    nn.backpropagation()
    for i in range(len(nn.weights)):
        nn.weights[i] -= nn.nabla_w_C[i]
        nn.bias[i] -= nn.nabla_b_C[i]
    nn.feed_forward()

plt.figure()
plt.plot(x, y, label="y")
plt.plot(x, nn.a[-1], label='NN pred')
plt.legend()
plt.show()
