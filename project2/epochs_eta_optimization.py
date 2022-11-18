# to test the optimization class
import os
from plot import *
from metrics import *
from optimization import *
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3155)
avec = np.array([0, 1, 1])
deg = len(avec)-1
x = np.linspace(-1,1,500)
# using simple linear regression to validate the gradient descent methods
X = np.ones((x.shape[0],deg+1))
for i in range(1,X.shape[1]):
    X[:,i] = x**i
y = test_func_poly_deg_p(deg = deg, avec = avec, x = x)[:,np.newaxis]
ynoisy = y + np.random.normal(size=y.shape)

# set the parameters
tol = 1e-8
epochs = 100
n_mini_batches = 100
max_iter = epochs*n_mini_batches
size_mini_batches = y.shape[0]//n_mini_batches
etas = np.logspace(-4,-1,4)
lmbda = 1e-3
labels = {'grad_desc' : 'GD','grad_desc_mom' : 'GD with momentum', 'adagrad' : 'Deterministic Adagrad',  'sgd' : 'SGD', 'rms_prop' : 'RMS propagation', 'adagrad_sgd' : 'Stochastic Adagrad', 'adam' : 'ADAM'}
det_methods = ['grad_desc', 'grad_desc_mom', 'adagrad']
stoch_methods = ['sgd', 'rms_prop','adagrad_sgd', 'adam']
gamma = 0.9 #1e-3
beta1 = 0.9 #1e-2
beta2 = 0.99 #1e-4
w_mom = True
for method in ['grad_desc_mom']:    
    mses = []
    for j in range(len(etas)):
        optimizer = optimizers(X, ynoisy, cost_Ridge, tol=tol, eta = etas[j], w_mom = w_mom, verbose=True)
        optimizer(method = method,epochs = epochs, size_mini_batches = size_mini_batches, max_iter = max_iter, 
                  store_mse = True, verbos = True, **{'lambda' : lmbda, 'scheme': 'linear', 't0' : 1e-1, 't1' : 1e-2})
        mses.append(optimizer.mse)
    # plot the mse vs epoch for given method and parameters
    fig = plt.figure()
    plt.title("Optimization methods learning rate")
    for i in range(len(etas)):
        # per epoch
        if method in stoch_methods:
            plt.plot(np.arange(epochs)+1, mses[i][:,-1], label="%s $\\eta = %.4f$"%(labels[method],etas[i]))
        if method in det_methods:
            plt.plot(np.arange(max_iter)+1, mses[i], label="%s $\\eta = %.4f$"%(labels[method],etas[i]))
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend()
    if method in stoch_methods:
        fig.savefig(os.getcwd() + "/plots/prob_a/epochs_etas_mse_%s_lambda_10%.2f_gamma_%.2f_beta1_%.2f_beta2_%.2f.png"%(method, np.log10(lmbda), gamma, beta1, beta2 ), dpi=150)
    if method in det_methods:
        fig.savefig(os.getcwd() + "/plots/prob_a/epochs_etas_mse_%s_lambda_10%.2f_gamma_%.2f.png"%(method, np.log10(lmbda), gamma ), dpi=150)

# scikit learn for comparison
#from sklearn.linear_model import SGDRegressor
#mses_skl = np.zeros((epochs))
#for i in range(epochs):
#    reg = SGDRegressor(max_iter=epochs, tol=tol, loss="squared_error", penalty='l2', 
#                       alpha = lmbda, learning_rate = 'constant', eta0 = eta)
#    reg.partial_fit(X, ynoisy.ravel())
#    mses_skl[i] = MSE(y, reg.predict(X))
#plt.plot(np.arange(epochs) + 1, mses_skl, label="MLPRegressor")
