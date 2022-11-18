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

# params common for NN and linear regression methods
lmbda  = 1e-3

# linear regression
X = np.c_[np.ones((n,1)), x, x**2]
XT_X = X.T @ X
#Ridge parameter lambda
Id = n*lmbda* np.eye(XT_X.shape[0])
# Hessian matrix to get a feel for the restriction on the learning rate
H = (2.0/n)* XT_X+2*lmbda* np.eye(XT_X.shape[0])
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")

# hyperparameters of the FFNN
eta = 1e-1
n_hidden_neurons = 10
# default value of size of mini batch for SGD in sklearn is min(200, nsamples)
epochs = 100
batch_size = 5
beta1 = 0.9
beta2 = 0.99
gamma = 0.9
w_mom = True
# predict and plot with regression algorithms
beta_ols = np.linalg.pinv(XT_X) @ X.T @ ynoisy
beta_ridge = np.linalg.inv(XT_X+Id) @ X.T @ ynoisy
ypred_ols = X @ beta_ols
ypred_ridge = X @ beta_ridge
fig, axs = plt.subplots(1,3,figsize=(15,5))
fig.suptitle("Comparison Linear Regression and FFNN with %i neurons, 1 hidden layer and \n $\mathrm{max}\\lambda_{H} = %.3f $ and $\\eta = 10^{%.2f}$"%(n_hidden_neurons, np.max(EigValues), np.log10(eta)) )
axs[0].scatter(x, ynoisy, label='Noisy y', color = 'c')

method = 'sgd'
for lmbda in [0,1e-5]:
    # Sklearn
    dnn = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons), activation='logistic', solver = method, momentum = 0.9,
                       alpha=lmbda, beta_1 = beta1, beta_2 = beta2, learning_rate = 'constant', learning_rate_init=eta, max_iter=epochs, batch_size = batch_size)
    dnn.fit(X_train, Y_train)
    axs[1].plot(np.arange(len(dnn.loss_curve_)) + 1, dnn.loss_curve_, label = 'SL loss $\\lambda = 10^{%.2f}$'%(np.log10(lmbda)) )
    # init NN and set up hidden and output layer
    nn = Neural_Network(X_train,  Y_train, costfunc = 'ridge', eta=eta, gamma = gamma, beta1 = beta1, beta2 = beta2, 
                        w_mom = w_mom, method = method, symbolic_differentiation = False)
    nn.add_layer(nodes = n_hidden_neurons, af = 'sigmoid')# weights = np.random.normal(0,.1,size = (X_train.shape[1], n_hidden_neurons) ) )
    nn.output_layer(af = 'linear')# weights = np.random.normal(0,.1,size = (n_hidden_neurons, X_train.shape[1]) ))
    # do SGD
    # epochs, mini batches
    nn.SGD(epochs, batch_size, printout = True, plot = False, **{'lambda' : lmbda})
    axs[1].plot(np.arange(len(nn.losses)), nn.losses, label = 'FFNN loss $\\lambda = 10^{%.2f}$'%(np.log10(lmbda)))
    axs[2].plot(np.arange(len(nn.l2norm_gradC_weights)) + 1, nn.l2norm_gradC_weights, label = 'FFNN $\\lambda = 10^{%.2f}$'%(np.log10(lmbda)))
    # plot for comparison
    axs[0].plot(x, nn.predict(X_train), label='Pred NN $\\lambda = 10^{%.2f}$'%(np.log10(lmbda)), marker = 'o', ms = 3)
    axs[0].plot(x, dnn.predict(X_train), label='sklearn MLPRegressor $\\lambda = 10^{%.2f}$'%(np.log10(lmbda)), ls='--')
axs[0].plot(x, ypred_ols, label = 'OLS lin reg', color = 'k', marker = '^', ms = 3, alpha = 0.4)
axs[0].plot(x, ypred_ridge, label = 'Ridge lin reg', color = 'grey', marker = 'd', ms = 3, alpha = 0.5 )
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Loss")
axs[2].set_xlabel("Epochs")
axs[2].set_ylabel("$\\Vert \\nabla_w C \\Vert_2$")
[ax.legend() for ax in axs]

# save figure
fig.savefig(os.getcwd() + "/plots/prob_b/simple_comparison_linreg_ffnn.png", dpi = 150)
