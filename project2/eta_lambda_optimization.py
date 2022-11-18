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

max_iter = 500
tol = 1e-4
lambdas = np.array(list(np.logspace(-5,1,5)) + [0])
etas = np.logspace(-5,-1,4)

# scikit learn for comparison
from sklearn.linear_model import SGDRegressor
mses_skl = np.zeros((len(lambdas),len(etas)))
for j in range(len(etas)):
    for i in range(len(lambdas)):
        reg = SGDRegressor(max_iter=max_iter, tol=tol, loss="squared_error", penalty='l2', alpha = lambdas[i], learning_rate = 'constant', eta0 = etas[j])
        reg.fit(X, ynoisy.ravel())
        mses_skl[i,j] = MSE(y, reg.predict(X))
ypred = reg.predict(X)
plt.plot(x, y, label="y")
plt.plot(x, ypred, label="MLPRegressor")
plt.legend()

plot_heatmap(etas,lambdas,mses_skl, title = 'MSE SGD sklearn', store = True,
             store_dir = os.getcwd() + "/plots/prob_a/eta_lambda_mses_sgd_sklearn_heatmap",
             xlabel = "$\\eta$", ylabel='$\\lambda$', cbar_label = 'MSE', vmin = np.min(mses_skl), vmax = np.max(mses_skl))

labels = {'sgd' : 'SGD', 'rms_prop' : 'RMS propagation', 'adagrad' : 'Adagrad', 'adam' : 'ADAM'}
methods = ['sgd', 'rms_prop','adagrad', 'adam']
gamma = 1e-3
beta1 = 0.9
beta2 = 0.99
plot_lines = True
for method in ['adam']:#methods:
    mses = np.zeros((len(lambdas),len(etas)))
    for j in range(len(etas)):
        for i in range(len(lambdas)):
            optimizer= optimizers(X, ynoisy, cost_Ridge, eta = etas[j], gamma = gamma, beta1 = beta1, beta2 = beta2, w_mom = True, verbose=True)
            optimizer(method = method, epochs = 200, size_mini_batches = 100, **{'lambda' : lambdas[i], 'scheme': 'linear', 't0' : 1e-1, 't1' : 1e-2})
            mses[i,j] = MSE(y, X @ optimizer.theta)
            if plot_lines:
                plt.plot(x,X @ optimizer.theta, label=labels[method], ls='--')
    plot_heatmap(etas,lambdas,mses, title = 'MSE %s'%labels[method], store = True, 
                 store_dir = os.getcwd() + "/plots/prob_a/eta_lambda_mses_%s_heatmap"%method,
                 xlabel = "$\\eta$", ylabel='$\\lambda$', cbar_label = 'MSE', vmin = np.min(mses), vmax = np.max(mses))



if plot_lines:
    plt.plot(x,y, color = 'k')
    plt.legend()
    plt.show()


