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

#Select features relevant to classification (texture,perimeter,compactness and symmetery) 
#and add to input matrix

temp1=np.reshape(x[:,1],(len(x[:,1]),1))
temp2=np.reshape(x[:,2],(len(x[:,2]),1))
X=np.hstack((temp1,temp2))      
temp=np.reshape(x[:,5],(len(x[:,5]),1))
X=np.hstack((X,temp))       
temp=np.reshape(x[:,8],(len(x[:,8]),1))
X=np.hstack((X,temp))       
# center and scale
X = (X - np.mean(X,axis=0, keepdims=True) )/np.std(X, axis=0)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1, random_state = 0)   #Split datasets into training and testing
# convert to onehot vectors
y_train=to_categorical_numpy(y_train)     #Convert labels to categorical when using categorical cross entropy
y_test=to_categorical_numpy(y_test)

del temp1,temp2,temp

print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)

eta= 1e-1               #Define vector of learning rates (parameter to SGD optimiser)
lmbda = 1e-5                   #Define vector of learning rates (parameter to SGD optimiser)
n_neurons = 5
n_hidden_layers = 2
epochs= 100                               #Number of reiterations over the input data
batch_size=8                               #Number of samples per gradient update
method = 'sgd'
beta1 = 0.9
beta2 = 0.99
fig, ax = plt.subplots()
fig.suptitle("Loss AFs %i neurons, %i layers $\\lambda = 10^{%.2f}, \\eta = 10^{%.2f}$"%(n_neurons, n_hidden_layers, np.log10(lmbda), np.log10(eta) ))
for af in [ 'relu','sigmoid', 'leaky_relu', 'tanh']:
    print(af)
    nn = Neural_Network(X_train,  y_train, costfunc = 'cross_entropy_l2reg', eta=eta, w_mom = True, beta1 = beta1, beta2 = beta2, method = method, symbolic_differentiation = True)
    for k in range(n_hidden_layers):
        if k == 0: nn.add_layer(nodes = n_neurons, af = af) 
        else: nn.add_layer(nodes = n_neurons, af = af)
        print(nn.nodes)
    print(nn.nodes)
    nn.add_layer(nodes = 2, af = 'sigmoid') 
    # do SGD
    nn.SGD(epochs = epochs, size_mini_batches = batch_size, printout=True,**{'lambda' : lmbda})
    print(nn.losses.shape)
    ax.plot(np.arange(len(nn.losses)) + 1, nn.losses, label = '%s'%af)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy with L2 regularization")
ax.legend()
fig.savefig(os.getcwd() + "/plots/prob_d/activation_functions_classifications_optimal_params.png")
