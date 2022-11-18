# to test the NN on a classification problem
from plot import plot_heatmap
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from neural_network import *
from test import test_func_poly_deg_p
from project1 import R2score, MSE
# for reproducability
np.random.seed(0)

# gernerate the simple 2deg poly
x = np.linspace(-1,1,500)
y = test_func_poly_deg_p(deg = 2, avec = [0,1,1], x = x)
ynoisy = y + np.random.normal(0,.1,size=y.shape)
x = (x - np.mean(x))/np.std(x) # pre-processing of input
X = x[:,np.newaxis]
Y = ynoisy[:,np.newaxis]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)


eta= 1e-1                    #Define vector of learning rates (parameter to SGD optimiser)
lmbda =1e-5                                  #Define hyperparameter
n_hidden_neurons = 10
epochs= 200                                 #Number of reiterations over the input data
batch_size= 10                              #Number of samples per gradient update

# %%

"""

Define function to return Deep Neural Network model

"""
   
# init NN
# building our neural network
method = 'sgd'
beta1 = 0.9
beta2 = 0.99
af = 'sigmoid'
w_mom = True
weightss = [np.random.normal(0,.1,size = (X_train.shape[1], n_hidden_neurons) ),
            np.random.normal(0,.5,size = (X_train.shape[1], n_hidden_neurons) ),
            np.random.normal(0, 1,size = (X_train.shape[1], n_hidden_neurons) ), 
            np.ones( (X_train.shape[1], n_hidden_neurons) )*0.5]
biases = [np.zeros((n_hidden_neurons,1)),
          np.ones((n_hidden_neurons,1))*0.01,
          np.ones((n_hidden_neurons,1))*0.1,
          np.ones((n_hidden_neurons,1))]
label_weights = ['$w_{ij}\\sim \\mathcal{N}(0,0.1)$','$w_{ij}\\sim \\mathcal{N}(0,0.5)$','$w_{ij}\\sim \\mathcal{N}(0,1)$','$w_{ij} = 0.5$']
label_bias = ['$b_i = 0$','$b_i = 0.01$','$b_i = 0.1$','$b_i = 1$']
for i in range(len(biases)):
    bias = biases[i]
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    fig.suptitle("Different initalizations $\\lambda = 10^{%.2f}, \\eta = 10{%.2f}, m = %i$ "%(np.log10(lmbda),np.log10(eta), batch_size ) + label_bias[i] )
    for j in range(len(weightss)):
        weights = weightss[j]
        nn = Neural_Network(X_train, Y_train, costfunc = 'ridge', eta=eta, w_mom = w_mom, beta1 = beta1, beta2 = beta2, method = method, symbolic_differentiation = True)
        nn.add_layer(nodes = n_hidden_neurons, af = 'sigmoid',  
                     weights = weights, bias = bias )        
        # output layer
        nn.add_layer(nodes = 2, af = 'sigmoid')
        # do SGD
        nn.SGD(epochs = epochs, size_mini_batches = batch_size, printout=True,**{'lambda' : lmbda})
        axs[0].plot(np.arange(len(nn.losses)), nn.losses, label = 'FFNN loss ' + label_weights[j])
        axs[1].plot(np.arange(len(nn.l2norm_gradC_weights)) + 1, nn.l2norm_gradC_weights, label = 'FFNN'  + label_weights[j])
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss: $\\Vert y - a \\Vert^2_2 - \\lambda \\Vert W \\Vert$")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("$\\Vert \\nabla_W C\\Vert_2$")
    [ax.legend() for ax in axs]
    fig.savefig(os.getcwd() + "/plots/prob_b/initalization_reg_bias_%.3f_FFNN.png"%bias[0], dpi=150)
