# a script that plots the comparison between the regression methods
# and the optimization methods
from optimization import *
from metrics import *
from plot import *
import os
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# the number of datapoints
n = 100
x = np.linspace(-1,1,n)[:,np.newaxis]
y = x + x**2 + np.random.normal(0, 0.5, size=x.shape)

X = np.c_[np.ones((n,1)), x, x**2]
print(X.shape)
XT_X = X.T @ X
print(XT_X.shape)
#Ridge parameter lambda
lmbda  = 0.001
Id = n*lmbda* np.eye(XT_X.shape[0])

# Hessian matrix
H = (2.0/n)* XT_X+2*lmbda* np.eye(XT_X.shape[0])
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")


beta_ols = np.linalg.pinv(XT_X) @ X.T @ y
beta_ridge = np.linalg.inv(XT_X+Id) @ X.T @ y
print(beta_ridge)
# Start plain gradient descent
beta = np.random.randn(3,1)

eta = 1.0/np.max(EigValues)
Niterations = 100

for iter in range(Niterations):
    gradients = 2.0/n*X.T @ (X @ (beta)-y)+2*lmbda*beta
    beta -= eta*gradients

print(beta)
ypredict = X @ beta
ypredict_ols = X @ beta_ols
ypredict_ridge = X @ beta_ridge
plot = False
if plot:
    plt.plot(x, ypredict, "r-", label="GD")
    plt.plot(x, ypredict_ols, "C2-", label="OLS", ls='--')
    plt.plot(x, ypredict_ridge, "b-", label="Ridge")
    plt.plot(x, y ,'ro')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Gradient descent example for Ridge')
    plt.legend()
    plt.show()

etas = np.logspace(-4,-1, 4)
lambdas = np.logspace(-5,1,7)
print("etas: ", etas, "lambdas : ", lambdas)
mses_ridge = np.zeros((len(lambdas)))
mses_optim = np.zeros((len(etas), len(lambdas)))
method = 'gd'
labels = {'gd' : 'GD', 'sgd' : 'SGD', 'rms_prop' : 'RMS propagation', 'adagrad' : 'Adagrad', 'adam' : 'ADAM'}

epochs = 100
n_mini_batches = 10
size_mini_batches = n//n_mini_batches
max_iter = epochs * n_mini_batches
beta1 = 1e-2
beta2 = 1e-4
gamma = 1e-3
for j in range(len(lambdas)):
    Id = n*lambdas[j]* np.eye(XT_X.shape[0])
    beta_ridge = np.linalg.inv(XT_X+Id) @ X.T @ y
    ypredict_ridge = X @ beta_ridge
    mses_ridge[j] = MSE(y, ypredict_ridge)
    for i in range(len(etas)):
        optimizer= optimizers(X, y, cost_Ridge, eta = etas[i], gamma = gamma, beta1 = beta1, beta2 = beta2, w_mom = True, verbose=True)
        optimizer(method = method, epochs = epochs, size_mini_batches = size_mini_batches, **{'lambda' : lambdas[j], 'scheme': 'linear', 't0' : 1e-1, 't1' : 1e-2})
        mses_optim[i,j] = MSE(y, X @ optimizer.theta)

plot_heatmap(etas,lambdas,mses_optim.T, title = 'MSE %s'%labels[method], store = True,
                 store_dir = os.getcwd() + "/plots/prob_a/eta_lambda_mses_%s_heatmap_ridge_comparison"%method,
                 xlabel = "$\\eta$", ylabel='$\\lambda$', cbar_label = 'MSE', vmin = np.min(mses_optim), vmax = np.max(mses_optim))


# plot ridge vs lambda
fig = plt.figure()
plt.plot(lambdas,mses_ridge, label='Ridge MSE')
plt.xlabel('$\\lambda$')
plt.ylabel('MSE')
fig.savefig(os.getcwd() + "/plots/prob_a/lambda_mses_ridge.png", dpi=150)

