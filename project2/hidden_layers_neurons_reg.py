# test cript for the NN
import os
from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
np.random.seed(3155)

# gernerate the simple 2deg poly
n = 500
x = np.linspace(-1,1,n)
y = test_func_poly_deg_p(deg = 2, avec = [0,1,1], x = x)
ynoisy = y + np.random.normal(0,.1,size=y.shape)
x = (x - np.mean(x))/np.std(x) # pre-processing of input
X_train = x[:,np.newaxis]
Y_train = y[:,np.newaxis]

# hyperparameters of the FFNN
lmbda  = 1e-3
eta = 5e-2
n_hidden_neurons = [[(1,), (1,1), (1,1,1)],
                    [(5,), (5,5), (5,5,5)],
                    [(10,), (10,10), (10,10,10)],
                    [(20,), (20,20), (20,20,20)],
                    [(30,), (30,30), (30,30,30)],
                    [(40,), (40,40), (40,40,40)]]
                    
# default value of size of mini batch for SGD in sklearn is min(200, nsamples)
epochs = 50
batch_size = 5
beta1 = 0.9
beta2 = 0.99
gamma = 0.9
method = 'sgd'
for net in n_hidden_neurons:
    print(net)
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle("Number of layers and neurons $\\lambda = 10^{%.2f}, \\eta = 10{%.2f}$"%(np.log10(lmbda),np.log10(eta) ) )
    for layers in net:
        print(layers)
        # Sklearn
        dnn = MLPRegressor(hidden_layer_sizes=layers, activation='logistic', solver = method,momentum = 0.9,
                           alpha=lmbda, beta_1 = beta1, beta_2 = beta2, learning_rate = 'constant', learning_rate_init=eta, max_iter=epochs, batch_size = batch_size)
        dnn.fit(X_train, Y_train)
        axs[1].plot(np.arange(len(dnn.loss_curve_)) + 1, dnn.loss_curve_, label = 'SL loss layers: ' + str(layers))
        # init NN and set up hidden and output layer
        nn = Neural_Network(X_train,  Y_train, costfunc = 'ridge', eta=eta, gamma = gamma, beta1 = beta1, beta2 = beta2, 
                            w_mom = True,method = method, symbolic_differentiation = True)
        for neurons in layers:
            nn.add_layer(nodes = neurons, af = 'sigmoid')# weights = np.random.normal(0,.1,size = (X_train.shape[1], n_hidden_neurons) ) )
        nn.output_layer(af = 'linear')# weights = np.random.normal(0,.1,size = (n_hidden_neurons, X_train.shape[1]) ))
        # do SGD
        # epochs, mini batches
        nn.SGD(epochs, batch_size, printout = True, plot = False, **{'lambda' : lmbda})
        axs[1].plot(np.arange(len(nn.losses)), nn.losses, label = 'FFNN train loss layers: ' + str(layers))
        axs[1].plot(np.arange(len(nn.losses_dev)), nn.losses_dev, label = 'FFNN dev loss layers: ' + str(layers))
        axs[2].plot(np.arange(len(nn.l2norm_gradC_weights)) + 1, nn.l2norm_gradC_weights, label = 'FFNN layers: ' + str(layers))
        # plot for comparison
        axs[0].plot(x, nn.predict(X_train), label='Pred NN layers: ' + str(layers) , marker = 'o', alpha = 0.5)
        axs[0].plot(x, dnn.predict(X_train), label='sklearn MLPRegressor layers: ' + str(layers) )
    axs[0].plot(x, y, label='True y', color = 'm')
    axs[0].scatter(x, ynoisy, label='Noisy y', color = 'c')
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("$L^2$-norm of the weights $\\Vert W \\Vert_2$")
    [ax.legend() for ax in axs]

    # save figure
    fig.savefig(os.getcwd() + "/plots/prob_b/layers_neurons_ffnn_%s_eta_%.3f.png"%( str(net), eta ), dpi = 150)
