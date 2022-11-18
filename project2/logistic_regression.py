# logistic regression script
from plot import plot_heatmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from neural_network import *
# for reproducability
np.random.seed(100)
# load mnist dataset
cancer = datasets.load_breast_cancer()

# define inputs and labels
inputs = cancer.data
outputs = cancer.target
labels=cancer.feature_names[0:30]

# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))

x=inputs      #Reassign the Feature and Label matrices to other variables
y=outputs

# Visualisation of dataset (for correlation analysis)
plot = False
if plot:
    plot_breast_scatter(x, y)

# Generate training and testing datasets

#Select features relevant to classification (texture,perimeter,compactness and symmetery) 
#and add to input matrix

temp1=np.reshape(x[:,1],(len(x[:,1]),1))
temp2=np.reshape(x[:,2],(len(x[:,2]),1))
X=np.hstack((temp1,temp2))
temp=np.reshape(x[:,5],(len(x[:,5]),1))
X=np.hstack((X,temp))

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.1, random_state = 100)   #Split datasets into training and testing
# scale and center
X_train = (X_train - np.mean(X_train,axis=0, keepdims=True) )/np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test,axis=0, keepdims=True) )/np.std(X_test, axis=0)
y_train = to_categorical_numpy(y_train)     #Convert labels to categorical when using categorical cross entropy
y_test  = to_categorical_numpy(y_test)
print("Y Train : ", y_train)
print("Y Test : ", y_test)
del temp1,temp2,temp

print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)

eta=1e-1                                    #Define vector of learning rates (parameter to SGD optimiser)
lamda=0.01                                  #Define hyperparameter
etas = np.logspace(-4,-1,4)
lambdas = np.logspace(-4,1,6)
epochs= 10
batch_size= 8                               #Number of samples per gradient update
mini_batches = y_train.shape[0]//batch_size                        #Number of reiterations over the input data
tol = 1e-4

Train_accuracy_own = np.zeros((len(etas), len(lambdas)))
Test_accuracy_own = np.zeros((len(etas), len(lambdas)))
train_acc_logreg = np.zeros((len(etas), len(lambdas)))
test_acc_logreg = np.zeros((len(etas), len(lambdas)))
# make training set on correct format
outputs_train = np.zeros(y_train.shape[0])
outputs_train += y_train[:,1]
for i in range(len(etas)):
    for j in range(len(lambdas)):
        eta = etas[i]
        lamda = lambdas[j]
        """

        Logistic regression


        Sklearn for comparison

        """

        logreg = LogisticRegression(penalty='l2', C=1/lamda, tol=tol, fit_intercept=True,random_state=100, solver='liblinear',max_iter=epochs)
        logreg.fit(X_train, outputs_train)
        logreg_pred_train = logreg.predict(X_train)
        logreg_pred = logreg.predict(X_test)
        ypred_train_logreg = to_categorical_numpy(logreg_pred_train.astype(int))
        ypred_test_logreg = to_categorical_numpy(logreg_pred.astype(int))
        train_acc_logreg[i,j] = accuracy(y = y_train, a = ypred_train_logreg)
        test_acc_logreg[i,j] = accuracy(y = y_test, a = ypred_test_logreg)
        """

        Own implementation

        """

        # setting up the logistic regression which is the same as 
        # a single layer perceptron with the sigmoid function as the activation function
        nn = Neural_Network(X_train,  y_train, costfunc = 'cross_entropy_l2reg', eta=eta, symbolic_differentiation = True)
        # using 2 nodes just because the data is transormed into onehot vectors
        nn.add_layer(nodes = 2, af = 'sigmoid', bias = np.zeros((2,1)) )
        nn.feed_forward()

        # using SGD for optimization
        nn.SGD(epochs = epochs, size_mini_batches = batch_size, tol=tol,plot = False, printout=True,**{'lambda' : lamda})

        # set data to test data and predict using weights computed with SGD
        aL_train= nn.predict(X_train)
        p2b_train = probs_to_binary(probabilities = aL_train)
        acc_train = accuracy(y = y_train, a = p2b_train)
        Train_accuracy_own[i,j] = acc_train

        # set data to test data and predict using weights computed with SGD
        aL_test = nn.predict(X_test)
        p2b_test = probs_to_binary(probabilities = aL_test)
        acc_test = accuracy(y = y_test, a = p2b_test)
        Test_accuracy_own[i,j] = acc_test

# plot heatmap for own implementation

plot_heatmap(etas,lambdas,Train_accuracy_own.T, title = 'Accuracy Train Own Logit Regression', store = True,
             store_dir = os.getcwd() + "/plots/prob_e/eta_lambda_acc_logitreg_heatmap",
             xlabel = "$\\eta$", ylabel='$\\lambda$', cbar_label = 'Accuracy', vmin = 0, vmax = 1)

plot_heatmap(etas,lambdas,Test_accuracy_own.T, title = 'Accuracy Test Own Logit Regression', store = True,
             store_dir = os.getcwd() + "/plots/prob_e/eta_lambda_acc_logitreg_heatmap",
             xlabel = "$\\eta$", ylabel='$\\lambda$', cbar_label = 'Accuracy', vmin = 0, vmax = 1)

# plot heatmap for sklearn

plot_heatmap(etas,lambdas,train_acc_logreg.T, title = 'Accuracy Train Sklearn Logit Regression', store = True,
             store_dir = os.getcwd() + "/plots/prob_e/eta_lambda_acc_sklearn_logitreg_heatmap",
             xlabel = "$\\eta$", ylabel='$\\lambda$', cbar_label = 'Accuracy', vmin = 0, vmax = 1)

plot_heatmap(etas,lambdas,test_acc_logreg.T, title = 'Accuracy Test Sklearn Logit Regression', store = True,
             store_dir = os.getcwd() + "/plots/prob_e/eta_lambda_acc_sklearn_logitreg_heatmap",
             xlabel = "$\\eta$", ylabel='$\\lambda$', cbar_label = 'Accuracy', vmin = 0, vmax = 1)



