# test cript for the NN
from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(3155)
x = np.linspace(-1, 1,5000)
# using simple linear regression to validate the gradient descent methods
#X = np.ones((x.shape[0],2))
#X[:,1] = x
y = test_func_poly_deg_p(deg = 2, avec = [0,1,1], x = x)
ynoisy = y + np.random.normal(0,.1,size=y.shape)
#x = (x - np.mean(x))/np.std(x) # pre-processing of input
nn = Neural_Network(x[:,np.newaxis],  y[:,np.newaxis], costfunc = 'ols')
nn.add_layer(nodes = 10, af = 'tanh')
nn.add_layer(nodes = 10, af = 'tanh')
nn.output_layer(af = 'linear')
nn.feed_forward()
nn.backpropagation()
for iter in range(int(5e3)):
    nn.feed_forward()
    nn.backpropagation()
    nn.update_weights(eta = 1e-1)
    print(np.mean(np.square(nn.target-nn.a[nn.layers])))
plt.plot(x, y, label='y')
plt.plot(x, nn.a[nn.layers], label='$a^L$ NN')
#plt.plot(x, nn.z[nn.layers], label='$z^L$ NN')
plt.legend()
plt.show()
