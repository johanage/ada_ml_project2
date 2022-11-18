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
size_mini_batches = y.shape[0]//n_mini_batches
max_iter = epochs*n_mini_batches
eta = 0.01
lambdas = np.logspace(-5,0,6)
gamma = 0.9
beta1 = 0.9
beta2 = 0.99
w_mom = True
# labels and methods list
labels = {'grad_desc' : 'GD','grad_desc_mom' : 'GD with momentum', 'adagrad' : 'Deterministic Adagrad', 
          'sgd' : 'SGD', 'rms_prop' : 'RMS propagation', 'adagrad_sgd' : 'Stochastic Adagrad', 'adam' : 'ADAM'}
stoch_methods = ['sgd', 'rms_prop','adagrad_sgd', 'adam']
det_methods = ['grad_desc', 'grad_desc_mom', 'adagrad']
for method in stoch_methods:
    mses = []
    fig, axs = plt.subplots(1, 2, figsize = (10,5))
    fig.suptitle("Optimization methods regularization $\\eta = %.3f$"%eta)
    for j in range(len(lambdas)):
        optimizer = optimizers(X, ynoisy, cost_Ridge, tol=tol, eta = eta, w_mom = w_mom, verbose=True)
        optimizer(method = method,epochs = epochs, size_mini_batches = size_mini_batches, 
                  store_mse = True, **{'lambda' : lambdas[j], 'scheme': 'linear', 't0' : 1e-1, 't1' : 1e-2})
        mses.append(optimizer.mse)
        axs[0].plot(x, X @ optimizer.theta, label='Pred $\\lambda = %.4f$'%(lambdas[i]) )

    axs[0].plot(x,y, color = 'k', label='OG data')
    axs[0].scatter(x,ynoisy, color = 'grey', label = 'Noisy OG data') 
    for i in range(len(lambdas)):
        # per epoch
        if method in stoch_methods:
            axs[1].plot(np.arange(epochs)+1, mses[i][:,-1], label="%s $\\lambda = %.4f$"%(labels[method],lambdas[i]))
        if method in det_methods:
            axs[1].plot(np.arange(max_iter)+1, mses[i], label="%s $\\lambda = %.4f$"%(labels[method],lambdas[i]))
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("MSE")
    axs[0].set_ylabel('y')
    axs[0].set_xlabel('x')
    [ax.legend() for ax in axs]
    if method in stoch_methods:
        fig.savefig(os.getcwd() + "/plots/prob_a/xyplot_epochs_lambdas_mse_%s_eta_10%.2f_gamma_%.2f_beta1_%.2f_beta2_%.2f.png"%(method, np.log10(eta), gamma, beta1, beta2 ), dpi=150)
    if method in det_methods:
        fig.savefig(os.getcwd() + "/plots/prob_a/xyplot_epochs_lambdas_mse_%s_eta_10%.2f_gamma_%.2f.png"%(method, np.log10(eta), gamma ), dpi=150)


# scikit learn for comparison
#from sklearn.linear_model import SGDRegressor
#mses_skl = np.zeros((epochs))
#for i in range(epochs):
#    reg = SGDRegressor(max_iter=epochs, tol=tol, loss="squared_error", penalty='l2', 
#                       alpha = lmbda, learning_rate = 'constant', eta0 = eta)
#    reg.partial_fit(X, ynoisy.ravel())
#    mses_skl[i] = MSE(y, reg.predict(X))
#plt.plot(np.arange(epochs) + 1, mses_skl, label="MLPRegressor")
