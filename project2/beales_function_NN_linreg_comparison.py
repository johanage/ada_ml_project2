# test cript for the NN
from project1 import FrankeFunction, create_X, plot_surface
import os
from neural_network import Neural_Network
from test import test_func_poly_deg_p, beales_func
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
np.random.seed(3155)

# gernerate the simple 2deg poly
n = 100
x = np.linspace(-4,4,n)
y = np.linspace(-4,4,n)
# preprocessing if range is outside 0 and 1
x = (x - np.mean(x))/np.std(x) # pre-processing of input
y = (y - np.mean(y))/np.std(y) # pre-processing of input
x, y = np.meshgrid(x, y)
X = np.c_[x.ravel(), y.ravel()]
z = beales_func(x,y)
znoisy = z + np.random.normal(size=z.shape)
plot_surf = False
if plot_surf:
    fig = plt.figure(); ax = fig.add_subplot(projection='3d')
    plot_surface(x,y, z, fig = fig, ax = ax)

# linear regression 
polydeg = 5
X_linreg = create_X(x.ravel(), y.ravel(), polydeg)
X_train, X_test, Y_train, Y_test = train_test_split(X_linreg,znoisy.ravel()[:,np.newaxis])

X_linreg_train = create_X(X_train[:,1].ravel(), X_train[:,2].ravel(), polydeg)
X_linreg_test = create_X(X_test[:,1].ravel(), X_test[:,2].ravel(), polydeg)
XT_X = X_linreg_train.T @ X_linreg_train
#Ridge parameter lambda
lmbda  = 1e-3
Id = n*lmbda* np.eye(XT_X.shape[0])

# Hessian matrix of ridge
H = (2.0/n)* XT_X+2*lmbda* np.eye(XT_X.shape[0])
# Get the eigenvalues
EigValues, EigVectors = np.linalg.eig(H)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")
beta_ols = np.linalg.pinv(XT_X) @ X_linreg_train.T @ Y_train
beta_ridge = np.linalg.inv(XT_X+Id) @ X_linreg_train.T @ Y_train
# hyperparameters of the FFNN
eta = 1e-3
n_neurons = 100
# default value of size of mini batch for SGD in sklearn is min(200, nsamples)
epochs = 50
batch_size = 100
beta1 = 0.9
beta2 = 0.99
gamma = 0.9
method = 'sgd'
af = 'tanh'
w_mom = True
fig, axs = plt.subplots(1,2,figsize=(10,5))
fig1, (ax_linreg, ax_sl, ax_nn) = plt.subplots(1, 3, figsize=(15,5), subplot_kw= {'projection': '3d'} )
# plot the solution surface
[plot_surface(x,y, z, fig = fig1, ax = ax) for ax in [ax_linreg, ax_sl, ax_nn]]
ax_linreg.set_title("Linear Regression methods OLS, Ridge")
ax_sl.set_title("Sklearn MLP Regressor")
ax_nn.set_title("FFNN implementation")
# plot the linreg solution
ax_linreg.scatter(X_test[:,1], X_test[:,2], X_test @ beta_ols, c =  X_test @ beta_ols, cmap = plt.cm.plasma, label = 'OLS linreg')
ax_linreg.scatter(X_test[:,1], X_test[:,2], X_test @ beta_ridge, c =  X_test @ beta_ridge, cmap = plt.cm.viridis, label = 'Ridge linreg')
fig.suptitle("Franke $\\lambda = 10^{%.2f}, \\eta = 10{%.2f}$"%(np.log10(lmbda),np.log10(eta) ) )
# Sklearn
dnn = MLPRegressor(hidden_layer_sizes=(n_neurons,), activation=af, solver = method,momentum = 0.9,
                   alpha=lmbda, beta_1 = beta1, beta_2 = beta2, learning_rate = 'constant', learning_rate_init=eta, max_iter=epochs, batch_size = batch_size)
dnn.fit(X_train, Y_train.ravel())
axs[0].plot(np.arange(len(dnn.loss_curve_)) + 1, dnn.loss_curve_, label = 'SL loss layers 1')
# init NN and set up hidden and output layer
nn = Neural_Network(X_train, Y_train, costfunc = 'ridge', eta=eta, gamma = gamma, beta1 = beta1, beta2 = beta2, 
                    w_mom = w_mom, method = method, symbolic_differentiation = False)
nn.add_layer(nodes = n_neurons, af = af,
             weights = np.random.normal(0,1e-1,size = (X_train.shape[1], n_neurons) ),
             bias = np.zeros((n_neurons, 1) ) )
nn.output_layer(af = 'linear')# weights = np.random.normal(0,.1,size = (n_hidden_neurons, X_train.shape[1]) ))
# do SGD
nn.SGD(epochs, batch_size, printout = True, plot = False, **{'lambda' : lmbda})
axs[0].plot(np.arange(len(nn.losses)), nn.losses, label = 'FFNN train loss layers 1 ')
axs[0].plot(np.arange(len(nn.losses)), nn.losses_dev, label = 'FFNN dev loss layers 1 ')
axs[1].plot(np.arange(len(nn.l2norm_gradC_weights)) + 1, nn.l2norm_gradC_weights, label = 'FFNN layers 1')
# plot for comparison
print(X_train[:,1].shape, nn.predict(X_train).shape)
#ax_nn.scatter(X_train[:,1], X_train[:,2],np.sum(nn.predict(X_train), axis=1), c = np.sum(nn.predict(X_train), axis=1), 
#              label='Train pred NN layers 1 ' , marker = 'o', alpha = 0.5, cmap = plt.cm.Greens)
ax_nn.scatter(X_test[:,1], X_test[:,2], np.sum(nn.predict(X_test)/n, axis=1), c = np.sum(nn.predict(X_test), axis=1),  
              label='Test pred NN layers 1 ' , marker = 'o', alpha = 0.5, cmap = plt.cm.Oranges)
ax_sl.scatter(X_train[:,1], X_train[:,2], dnn.predict(X_train), c = dnn.predict(X_train), 
              label='Train sklearn MLPRegressor layers 1 ', cmap = plt.cm.Greys)
ax_sl.scatter(X_test[:,1], X_test[:,2], dnn.predict(X_test), c = dnn.predict(X_test),
              label='Test sklearn MLPRegressor layers 1 ', cmap = plt.cm.Purples )
for ax in [ax_linreg, ax_nn, ax_sl]:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel(" $\\Vert \\nabla_W C \\Vert_2$")
[ax.legend() for ax in axs]

# save figure
fig.savefig(os.getcwd() + "/plots/prob_b/beales_linreg_ffnn_comparison_eta_%.3f.png"%(eta ), dpi = 150)
fig1.savefig(os.getcwd() + "/plots/prob_b/beales_surface_linreg_ffnn_comparison_eta_%.3f.png"%(eta ), dpi = 150)
