# to test the optimization class
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
plt.plot(x, ynoisy, label='orig data')

max_iter = 20000
epochs = 100
size_mini_batches = 10
tol = 1e-4
# scikit learn for comparison
from sklearn.linear_model import SGDRegressor
reg = SGDRegressor(max_iter=max_iter, tol=tol, loss="squared_error", penalty='l2', alpha = 1e-3, learning_rate = 'constant', eta0 = 1e-2)
reg.fit(X, ynoisy.ravel())
ypred = reg.predict(X)
plt.plot(x, y, label="y")
plt.plot(x, ypred, label="MLPRegressor")
plt.legend()

optimizer = optimizers(X, ynoisy, cost_OLS, eta=1e-1, max_iter = 300)
optimizer(method = 'grad_desc')
plt.plot(x, X @ optimizer.theta, label="pred GD")
optimizer(method = 'grad_desc_mom')
plt.plot(x, X @ optimizer.theta, label="pred GD with momentum")

optimizer = optimizers(X, ynoisy, cost_OLS, eta=1e-1, max_iter = 300)
optimizer(method = 'sgd', epochs = epochs,size_mini_batches = size_mini_batches, **{'lambda' : 1e-3,'scheme': 'linear', 't0' : 1e-1, 't1' : 1e-2})
plt.plot(x,X @ optimizer.theta, label="pred SGD")
plt.legend()

optimizer = optimizers(X, ynoisy, cost_OLS, eta = 1e-1,w_mom = True, verbose=True)
optimizer(method = 'adagrad_sgd', epochs = epochs, size_mini_batches = size_mini_batches, **{'scheme': 'linear', 't0' : 1e-1, 't1' : 1e-2})
plt.plot(x,X @ optimizer.theta, label="Adagrad")
plt.legend()

optimizer = optimizers(X, ynoisy, cost_OLS, eta = 1e-1,w_mom = True, verbose=True, gamma = 1e-3)
optimizer(method = 'rms_prop', epochs = epochs,size_mini_batches = size_mini_batches, **{'scheme': 'linear', 't0' : 1e-1, 't1' : 1e-2})
plt.plot(x,X @ optimizer.theta, label="RMS propagation")

optimizer = optimizers(X, ynoisy, cost_OLS, eta=1e-1, max_iter = 300)
optimizer(method='adam', epochs = epochs,size_mini_batches = size_mini_batches, **{'lambda' : 0})
plt.plot(x,X @ optimizer.theta, label="ADAM")
plt.legend()
plt.show()


