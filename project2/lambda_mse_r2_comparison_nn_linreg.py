# test cript for the NN
import os
from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from project1 import MSE, R2score

np.random.seed(3155)

# gernerate the simple 2deg poly
n = 500
x = np.linspace(-1,1,n)
y = test_func_poly_deg_p(deg = 2, avec = [0,1,1], x = x)
ynoisy = y + np.random.normal(0,.1,size=y.shape)
x = (x - np.mean(x))/np.std(x) # pre-processing of input
X = x[:,np.newaxis]
Y = y[:,np.newaxis]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3155)
print("shapes of xtrain, xtest, ytrain, ytest", X_train.shape, X_test, shape, Y_train.shape, Y_test.shape)
# params common for NN and linear regression methods
lmbda  = 1e-3

# linear regression
n_train, n_test = len(X_train.ravel()), len(X_test.ravel())
X_linreg_train = np.c_[np.ones((n_train,1)), X_train.ravel(), X_train.ravel()**2]
X_linreg_test = np.c_[np.ones((n_test,1)), X_test.ravel(), X_test.ravel()**2]
XT_X = X_linreg_train.T @ X_linreg_train
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
method = 'sgd'
# predict and plot with regression algorithms
beta_ridge = np.linalg.inv(XT_X+Id) @ X_linreg_train.T @ Y_train.ravel()
ypred_ridge = X_linreg_test @ beta_ridge
fig, axs = plt.subplots(1,2,figsize=(10,5))
fig.suptitle("Regularization comparison Linear Regression and FFNN with %i neurons, 1 hidden layer and \n $\mathrm{max}\\lambda_{H} = %.3f $ and $\\eta = 10^{%.2f}$"%(n_hidden_neurons, np.max(EigValues), np.log10(eta)) )

lambdas = np.logspace(-5,0,10)
mse_nn =  np.zeros(len(lambdas))
mse_sl =  np.zeros(len(lambdas))
mse_ridge =  np.zeros(len(lambdas))
R2_nn =  np.zeros(len(lambdas))
R2_sl =  np.zeros(len(lambdas))
R2_ridge = np.zeros(len(lambdas))
i = 0
for lmbda in lambdas:
    # Sklearn MLP Reg
    dnn = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons), activation='logistic', solver = method, momentum = 0.9,
                       alpha=lmbda, beta_1 = beta1, beta_2 = beta2, learning_rate = 'constant', learning_rate_init=eta, max_iter=epochs, batch_size = batch_size)
    dnn.fit(X_train, Y_train)
    # implemented FFNN
    nn = Neural_Network(X_train,  Y_train, costfunc = 'ridge', eta=eta, gamma = gamma, beta1 = beta1, beta2 = beta2, 
                        w_mom = w_mom, method = method, symbolic_differentiation = False)
    nn.add_layer(nodes = n_hidden_neurons, af = 'sigmoid')# weights = np.random.normal(0,.1,size = (X_train.shape[1], n_hidden_neurons) ) )
    nn.output_layer(af = 'linear')# weights = np.random.normal(0,.1,size = (n_hidden_neurons, X_train.shape[1]) ))
    # sgd
    nn.SGD(epochs, batch_size, printout = True, plot = False, **{'lambda' : lmbda})
    # store metrics
    mse_nn[i] = MSE(Y_test, nn.predict(X_test))
    mse_sl[i] = MSE(Y_test, dnn.predict(X_test))
    mse_ridge[i] = MSE(Y_test, ypred_ridge)
    R2_nn[i] = R2score(Y_test, nn.predict(X_test))
    R2_sl[i] = R2score(Y_test, dnn.predict(X_test))
    R2_ridge[i] = R2score(Y_test, ypred_ridge)
 
    i+=1
axs[0].plot(lambdas,mse_ridge/np.max(mse_ridge), label = 'MSE Ridge linreg', color = 'm', marker = 'd', ms = 3, alpha = 0.5 )
axs[0].plot(lambdas,mse_nn/np.max(mse_nn), label = 'MSE NN', color = 'c', marker = 'd', ms = 3, alpha = 0.5 )
axs[0].plot(lambdas,mse_sl/np.max(mse_sl), label = 'MSE SL MLP Reg', color = 'y', marker = '^', ms = 3, alpha = 0.5 )
axs[1].plot(lambdas,R2_ridge/np.max(R2_ridge), label = '$R^2$-score Ridge linreg', color = 'm', marker = 'd', ms = 3, alpha = 0.5 )
axs[1].plot(lambdas,R2_nn/np.max(R2_nn), label = '$R^2$-score NN', color = 'c', marker = 'd', ms = 3, alpha = 0.5 )
axs[1].plot(lambdas,R2_sl/np.max(R2_sl), label = '$R^2$-score SL MLP Reg', color = 'y', marker = '^', ms = 3, alpha = 0.5 )
[ax.set_xlabel("$\\lambda$") for ax in axs]
axs[0].set_ylabel("MSE")
axs[0].set_ylabel("$R^2$-score")
[ax.legend() for ax in axs]

# save figure
fig.savefig(os.getcwd() + "/plots/prob_b/simple_comparison_linreg_ffnn.png", dpi = 150)
