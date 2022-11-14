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
epochs = 400
n_mini_batches = np.array([10,25,50,75,100])
size_mini_batches = y.shape[0]//n_mini_batches
eta = 1e-2
lmbda = 1e-3
# scikit learn for comparison
from sklearn.linear_model import SGDRegressor
mses_skl = np.zeros((epochs))
for i in range(epochs):
    reg = SGDRegressor(max_iter=epochs, tol=tol, loss="squared_error", penalty='l2', 
                       alpha = lmbda, learning_rate = 'constant', eta0 = eta)
    reg.partial_fit(X, ynoisy.ravel())
    mses_skl[i] = MSE(y, reg.predict(X))
#plt.plot(np.arange(epochs) + 1, mses_skl, label="MLPRegressor")

mses = []
method = 'sgd'
for j in range(len(mini_batches)):
    optimizer = optimizers(X, ynoisy, cost_Ridge, tol=tol, eta = eta, w_mom = False, verbose=True)
    optimizer(method = method,epochs = epochs, size_mini_batches = size_mini_batches[j], 
              store_mse = True, **{'lambda' : lmbda, 'scheme': 'linear', 't0' : 1e-1, 't1' : 1e-2})
    mses.append(optimizer.mse)

fig = plt.figure()
for i in range(len(mini_batches)):
    # per epoch
    plt.plot(np.arange(epochs)+1, mses_sgd[i][:,-1], label="SGD %i mini batches"%(n_mini_batches[i]))
    # per iteration
    #plt.plot(np.arange(len(mses_sgd[i].ravel()))+1, mses_sgd[i].ravel(), label="SGD %i mini batches"%(mini_batches[i]))
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()
fig.savefig(os.getcwd() + "/plots/prob_a/epochs_mini_batches_mse_%s.png"%method, dpi=150)
