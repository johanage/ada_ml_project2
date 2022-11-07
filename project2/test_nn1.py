# test cript for the NN
from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(3155)
x = np.linspace(-1,1,5000)
y = test_func_poly_deg_p(deg = 2, avec = [0,1,1], x = x)
ynoisy = y + np.random.normal(0,.1,size=y.shape)
x = (x - np.mean(x))/np.std(x) # pre-processing of input
# init NN
nn = Neural_Network(x[:,np.newaxis],  y[:,np.newaxis], costfunc = 'ols', eta=5e-2)
nn.add_layer(nodes = 5, af = 'sigmoid')
nn.add_layer(nodes = 10, af = 'leaky_relu')
nn.output_layer(af = 'linear')
# do SGD
# epochs, mini batches
nn.SGD(200, 200)
# set data to original and predict using weights computed with SGD
nn.target = nn.Ydata_full
nn.a[0] = nn.Xdata_full
nn.feed_forward()
plt.plot(x, y, label='y')
plt.plot(x, nn.a[nn.layers], label='$a^L$ NN')
#plt.plot(x, nn.z[nn.layers], label='$z^L$ NN')
plt.legend()
plt.show()
