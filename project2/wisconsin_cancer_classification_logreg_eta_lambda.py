# to test the NN on a classification problem
from plot import plot_heatmap
#import tensorflow as tf
#from tensorflow.keras.layers import Input
#from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
#from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
#from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
#from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
#from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from neural_network import *
# for reproducability
np.random.seed(0)
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
temp=np.reshape(x[:,8],(len(x[:,8]),1))
X=np.hstack((X,temp))       
# scale and center
X = (X - np.mean(X,axis=0, keepdims=True) )/np.std(X, axis=0)



X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1, random_state = 0)   #Split datasets into training and testing
y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]
del temp1,temp2,temp

print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)

eta= np.logspace(-3,-1,3)                   #Define vector of learning rates (parameter to SGD optimiser)
lambdas=np.array( [0] + list(np.logspace(-5,-1,3)))                   #Define vector of learning rates (parameter to SGD optimiser)
n_neurons = 1
n_hidden_layers = 0
epochs= 50                               #Number of reiterations over the input data
batch_size= 10                               #Number of samples per gradient update

# %%

"""

Define function to return Deep Neural Network model

"""
   
# init NN
# building our neural network
n_inputs, n_features = X_train.shape
Train_accuracy_own = np.zeros((len(lambdas),len(eta)))      #Define matrices to store accuracy scores as a function
Test_accuracy_own  = np.zeros((len(lambdas),len(eta)))       #of learning rate and number of hidden neurons for 
method = 'sgd'
gamma = 0.9
beta1 = 0.9
beta2 = 0.99
af = 'sigmoid'
for i in range(len(lambdas)):     #run loops over hidden neurons and learning rates to calculate 
    lmbda = lambdas[i]
    for j in range(len(eta)):      #accuracy scores 
        nn = Neural_Network(X_train,  y_train, costfunc = 'cross_entropy_l2reg', eta=eta[j], w_mom = True, 
                            gamma = gamma, beta1 = beta1, beta2 = beta2, method = method, symbolic_differentiation = True)
        nn.add_layer(nodes = 1, af = 'sigmoid')#, 
        #             weights = np.random.normal(0,1,size=(n_neurons, 2) ), 
        #             bias = np.random.normal(0,1,(2,1) ) )
        # do SGD
        nn.SGD(epochs = epochs, size_mini_batches = batch_size, printout=True,**{'lambda' : lmbda})

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
store_dir_train = "/plots/prob_d/logreg_lambdas_eta_heatmap_breast_train_%iepochs"%epochs
store_dir_test = "/plots/prob_d/logreg_lambdas_eta_heatmap_breast_test_%iepochs"%epochs

if method == 'adam':
    store_dir_train = store_dir_train + "adam_%s_beta1_1e%.2f_beta2_1e%.2f"%(af,beta1, beta2)
    store_dir_test = store_dir_test + "adam_%s_beta1_1e%.2f_beta2_1e%.2f"%(af,beta1, beta2)

plot_heatmap(eta,lambdas,Train_accuracy_own, title = 'Train neurons = %i layers = %i'%(n_neurons,n_hidden_layers),
             xlabel = '$\\eta$', ylabel = '$\\lambda$', type_axis = 'log',
             store = True, store_dir = os.getcwd() + store_dir_train)
plot_heatmap(eta,lambdas,Test_accuracy_own, title = 'Test neurons = %i layers = %i'%(n_neurons,n_hidden_layers), 
             xlabel = '$\\eta$', ylabel = '$\\lambda$', type_axis = 'log',
             store = True, store_dir = os.getcwd() + store_dir_test)

