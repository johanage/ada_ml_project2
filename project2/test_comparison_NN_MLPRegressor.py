# test cript for the NN
from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
np.random.seed(3155)

# gernerate the simple 2deg poly
x = np.linspace(-1,1,10000)
y = test_func_poly_deg_p(deg = 2, avec = [0,1,1], x = x)
ynoisy = y + np.random.normal(0,.1,size=y.shape)
x = (x - np.mean(x))/np.std(x) # pre-processing of input
X_train = x[:,np.newaxis]
Y_train = y[:,np.newaxis]

eta = 1e-1
n_hidden_neurons = 50
# default value of size of mini batch for SGD in sklearn is min(200, nsamples)
epochs = 400
batch_size = 75
dnn = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                   alpha=0, learning_rate = 'constant', learning_rate_init=eta, max_iter=epochs, batch_size = batch_size)
dnn.fit(X_train, Y_train)

# init NN and set up hidden and output layer
nn = Neural_Network(X_train,  Y_train, costfunc = 'ols', eta=eta, specify_grad = True)
nn.add_layer(nodes = n_hidden_neurons, af = 'sigmoid')
nn.output_layer(af = 'linear')
#for i in range(10000):
#    nn.feed_forward()
#    nn.backpropagation()
#    nn.update_weights(1)

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
