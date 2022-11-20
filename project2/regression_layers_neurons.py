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
np.random.seed(3155)

# gernerate the simple 2deg poly
x = np.linspace(-1,1,5000)
y = test_func_poly_deg_p(deg = 2, avec = [0,1,1], x = x)
ynoisy = y + np.random.normal(0,.1,size=y.shape)
#x = (x - np.mean(x))/np.std(x) # pre-processing of input
X = x[:,np.newaxis]
Y = ynoisy[:,np.newaxis]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3155)

eta= 1e-1                    #Define vector of learning rates (parameter to SGD optimiser)
lmbda =1e-5                                  #Define hyperparameter
n_hidden_neurons = [[(1,), (1,1), (1,1,1)],
                    [(5,), (5,5), (5,5,5)],
                    [(10,), (10,10), (10,10,10)],
                    [(20,), (20,20), (20,20,20)],
                    [(30,), (30,30), (30,30,30)],
                    [(40,), (40,40), (40,40,40)],
                    [(50,), (50,50), (50,50,50)],
                    [(100,), (100,100), (100,100,100)],
                    [(500,), (500,500), (500,500,500)]]

epochs= 100                                 #Number of reiterations over the input data
batch_size= 5                              #Number of samples per gradient update
w_mom = True
# %%

"""

Define function to return Deep Neural Network model

"""
   
# init NN
# building our neural network
Train_mse_own = np.zeros((len(n_hidden_neurons),len(n_hidden_neurons[0])))      #Define matrices to store accuracy scores as a function
Train_R2_own = np.zeros((len(n_hidden_neurons),len(n_hidden_neurons[0])))      #Define matrices to store accuracy scores as a function
Test_mse_own  = np.zeros((len(n_hidden_neurons),len(n_hidden_neurons[0])))      #of learning rate and number of hidden neurons for 
Test_R2_own  = np.zeros((len(n_hidden_neurons),len(n_hidden_neurons[0])))      #of learning rate and number of hidden neurons for 
method = 'sgd'
beta1 = 0.9
beta2 = 0.99
af = 'sigmoid'
i = 0
for net in n_hidden_neurons:
    print(net)
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    fig.suptitle("Number of layers and neurons %s with $\\lambda = 10^{%.2f}, \\eta = 10{%.2f}, m = %i$"%(str(net), np.log10(lmbda),np.log10(eta), batch_size ) )
    j = 0
    for layers in net:

        nn = Neural_Network(X_train, Y_train, costfunc = 'ridge', eta=eta, w_mom = w_mom, beta1 = beta1, beta2 = beta2, method = method, symbolic_differentiation = False)
        for neurons in layers:
            nn.add_layer(nodes = neurons, af = 'sigmoid')# weights = np.random.normal(0,.1,size = (X_train.shape[1], n_hidden_neurons) ) )        

        # output layer
        nn.add_layer(nodes = 2, af = 'sigmoid')#  weights = np.random.normal(0,1,size=(neurons, 2) )) 
        # do SGD
        print("Epochs: ", epochs, " # mini batch size :", batch_size)
        nn.SGD(epochs = epochs, size_mini_batches = batch_size, printout=True,**{'lambda' : lmbda})
        axs[0].plot(np.arange(len(nn.losses)), nn.losses, label = 'FFNN loss layers: ' + str(layers))
        axs[0].plot(np.arange(len(nn.losses)), nn.mse, label = 'FFNN MSE layers: ' + str(layers))
        axs[0].plot(np.arange(len(nn.losses)), nn.r2, label = 'FFNN R2 layers: ' + str(layers))
        axs[1].plot(np.arange(len(nn.l2norm_gradC_weights)) + 1, nn.l2norm_gradC_weights, label = 'FFNN layers: ' + str(layers))
        Train_mse_own[i,j] = MSE(nn.target, nn.a[nn.layers])
        Train_R2_own[i,j] = R2score(nn.target, nn.a[nn.layers])
        Test_mse_own[i,j] = MSE(Y_test, nn.predict(X_test))
        Test_R2_own[i,j] = R2score(Y_test, nn.predict(X_test))

        j += 1
    i += 1
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss: $\\Vert y - a \\Vert^2_2 - \\lambda \\Vert W \\Vert$")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("$\\Vert \\nabla_W C\\Vert_2$")
    [ax.legend() for ax in axs]
    fig.savefig(os.getcwd() + "/plots/prob_b/layers_neurons_%s_reg_FFNN.png"%(str(net)), dpi=150)
    
store_dir_train = "/plots/prob_b/%s_neurons_layers_heatmap_reg_train_"%method
store_dir_test = "/plots/prob_b/%s_neurons_layers_heatmap_reg_test_"%method

lneurons = np.array([1, 5, 10, 20, 30, 40, 50, 100, 500])
llayers = np.array([1,2,3,])
plot_heatmap(llayers,lneurons,Train_mse_own, title = 'Train MSE', type_axis = 'int',
             xlabel="Layers", ylabel="Neurons",
             store = True, store_dir = os.getcwd() + store_dir_train)
plot_heatmap(llayers,lneurons,Test_mse_own, title = 'Test MSE', type_axis = 'int', 
             xlabel="Layers", ylabel="Neurons",
             store = True, store_dir = os.getcwd() + store_dir_test)

plot_heatmap(llayers,lneurons,Train_R2_own, title = 'Train R2-score', type_axis = 'int',
             xlabel="Layers", ylabel="Neurons",
             store = True, store_dir = os.getcwd() + store_dir_train)
plot_heatmap(llayers,lneurons,Test_R2_own, title = 'Test R2-score', type_axis = 'int',
             xlabel="Layers", ylabel="Neurons",
             store = True, store_dir = os.getcwd() + store_dir_test)
