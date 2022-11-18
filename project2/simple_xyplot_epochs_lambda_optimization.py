# to test the optimization class
import os
from plot import *
from metrics import *
from optimization import *
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(3155)
x = np.linspace(-1,1,500)
y = (x + x**2)[:,np.newaxis]
ynoisy = y + np.random.normal(0,0.1,size=y.shape)
deg = 2
X = np.ones((x.shape[0],deg+1))
for i in range(1,X.shape[1]):
    X[:,i] = x**i

#x = (x - np.mean(x, axis=0))/np.std(x, axis=0)
# set the parameters
tol = 1e-8
epochs = 200
n_mini_batches = 100
size_mini_batches = ynoisy.shape[0]//n_mini_batches
max_iter = epochs*n_mini_batches
eta = 0.01
lambdas = [1e-2]
gamma = 0.9
beta1 = 0.9
beta2 = 0.99
w_mom = True
# labels and methods list
labels = {'grad_desc' : 'GD','grad_desc_mom' : 'GD with momentum', 'adagrad' : 'Deterministic Adagrad', 
          'sgd' : 'SGD', 'rms_prop' : 'RMS propagation', 'adagrad_sgd' : 'Stochastic Adagrad', 'adam' : 'ADAM'}
stoch_methods = ['sgd', 'rms_prop','adagrad_sgd', 'adam']
det_methods = ['grad_desc', 'grad_desc_mom', 'adagrad']
for method in ['sgd']:
    mses = []
    fig, axs = plt.subplots(1, 2, figsize = (10,5))
    fig.suptitle("Optimization methods regularization $\\eta = %.3f$"%eta)
    axs[0].plot(x,y, color = 'k', label='OG data')
    axs[0].scatter(x,ynoisy, color = 'grey', label = 'Noisy OG data') 
    for j in range(len(lambdas)):
        optimizer = optimizers(X, ynoisy, cost_Ridge, tol=tol, eta = eta, w_mom = w_mom, verbose=True)
        optimizer(method = method,epochs = epochs, size_mini_batches = size_mini_batches, 
                  store_mse = True, **{'lambda' : lambdas[j], 'scheme': 'linear', 't0' : 1e-1, 't1' : 1e-2})
        mses.append(optimizer.mse)
        axs[0].plot(x, X @ optimizer.theta, label='Pred $\\lambda = %.4f$'%(lambdas[j]) )
        # per epoch
        if method in stoch_methods:
            axs[1].plot(np.arange(epochs)+1, mses[j][:,-1], label="%s $\\lambda = %.4f$"%(labels[method],lambdas[j]))
        if method in det_methods:
            axs[1].plot(np.arange(max_iter)+1, mses[j], label="%s $\\lambda = %.4f$"%(labels[method],lambdas[j]))
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("MSE")
    axs[0].set_ylabel('y')
    axs[0].set_xlabel('x')
    [ax.legend() for ax in axs]
    if method in stoch_methods:
        fig.savefig(os.getcwd() + "/plots/prob_a/xyplot_epochs_lambda_%.3f_mse_%s_eta_10%.2f_gamma_%.2f_beta1_%.2f_beta2_%.2f.png"%(lambdas[0], method, np.log10(eta), gamma, beta1, beta2 ), dpi=150)
    if method in det_methods:
        fig.savefig(os.getcwd() + "/plots/prob_a/xyplot_epochs_lambda_%.3f_mse_%s_eta_10%.2f_gamma_%.2f.png"%(lambdas[0], method, np.log10(eta), gamma ), dpi=150)

