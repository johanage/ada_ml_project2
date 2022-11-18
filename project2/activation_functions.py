# test cript for the NN
import os
from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(155)

# gernerate the simple 2deg poly
x = np.linspace(-1,1,500)
y = test_func_poly_deg_p(deg = 2, avec = [0,1,1], x = x)
ynoisy = y + np.random.normal(0,.1,size=y.shape)
X_train = x[:,np.newaxis]
Y_train = y[:,np.newaxis]
# center and scale
#X_train = (X_train - np.mean(X_train, axis = 0))/np.std(X_train, axis=0)
eta = 1e-1
n_hidden_neurons = 25
# default value of size of mini batch for SGD in sklearn is min(200, nsamples)
epochs = 400
batch_size = 10
lmbda = 1e-3
beta1 = 0.9
beta2 = 0.99
gamma = 0.9
method = 'sgd'
lambdas = np.array([0] + list(np.logspace(-4,0,5)) )
w_mom = True
# init NN and set up hidden and output layer
afs = {'Sigmoid' : 'sigmoid', 'ReLU' : 'relu', 'Leaky ReLU' : 'leaky_relu'}
markers = {'Sigmoid' : 'o', 'ReLU' : '^', 'Leaky ReLU' : 'd'}

fig, axs = plt.subplots(1,2,figsize=(10,5))
for key, value in afs.items():
    nn = Neural_Network(X_train,  Y_train, costfunc = 'ridge', eta=eta, gamma = gamma, beta1 = beta1, beta2 = beta2, 
                        method = method, symbolic_differentiation = False, w_mom = w_mom)
    nn.add_layer(nodes = n_hidden_neurons, af = value, weights = np.random.normal(0,.5,size=(X_train.shape[1], n_hidden_neurons)) )
    nn.output_layer(af = 'linear', weights = np.random.normal(0,.5,size=(n_hidden_neurons, X_train.shape[1])) )
    # do SGD
    # epochs, mini batches
    nn.SGD(epochs, batch_size, printout = True, plot = False, **{'lambda' : lambdas[2]})
    # plot for comparison
    axs[0].plot(x, nn.predict(X_train), label='Pred NN %s'%value, marker = markers[key], alpha = 0.5)
    axs[1].plot(np.arange(nn.losses.shape[0])+1, nn.losses, label = '%s'%value, marker = markers[key])
axs[0].plot(x, y, label='y')
axs[0].set_xlabel('$x$')
axs[0].set_ylabel('$y$')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel(r'$C(\vec{ \theta} )$')
[ax.legend() for ax in axs]
fig.savefig(os.getcwd() + "/plots/prob_c/activation_functions_analysis_%i_epochs_%s_wmom_%s.png"%(epochs, method, str(w_mom)), dpi=150)
