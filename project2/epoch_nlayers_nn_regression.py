# test cript for the NN
from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(3155)

# gernerate the simple 2deg poly
x = np.linspace(-1,1,500)
y = test_func_poly_deg_p(deg = 2, avec = [0,1,1], x = x)
ynoisy = y + np.random.normal(0,.1,size=y.shape)
x = (x - np.mean(x))/np.std(x) # pre-processing of input
X_train = x[:,np.newaxis]
Y_train = y[:,np.newaxis]

eta = 1e-1
n_hidden_neurons = 50
epochs = 100
batch_size = 32
lambdas = np.logspace(-4,0,5)
# init NN and set up hidden and output layer
nn = Neural_Network(X_train,  Y_train, costfunc = 'ridge', eta=eta, symbolic_differentiation = True)
nn.add_layer(nodes = n_hidden_neurons, af = 'sigmoid')
nn.output_layer(af = 'linear')
# do SGD
# epochs, mini batches
nn.SGD(epochs, batch_size, printout = True, plot = True, **{'lambda' : 0})
# plot for comparison
plt.plot(x, nn.predict(X_train), label='$a^L$ NN', marker = 'o', alpha = 0.5)
plt.plot(x, dnn.predict(X_train), label='sklearn MLPRegressor')
plt.plot(x, y, label='y')
plt.plot(x, nn.z[nn.layers], label='$z^L$ NN')
plt.legend()
plt.show()
