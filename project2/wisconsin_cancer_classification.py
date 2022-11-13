# to test the NN on a classification problem
from plot import plot_heatmap
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os 
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
temp=np.reshape(x[:,8],(len(x[:,8]),1))
X=np.hstack((X,temp))       


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1, random_state = 100)   #Split datasets into training and testing
# scale and center
X_train = (X_train - np.mean(X_train,axis=0, keepdims=True) )/np.std(X_train, axis=0)
y_train=to_categorical_numpy(y_train)     #Convert labels to categorical when using categorical cross entropy
y_test=to_categorical_numpy(y_test)

del temp1,temp2,temp

print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)

# %%

# Define tunable parameters"

eta=np.logspace(-3,-1,3)                    #Define vector of learning rates (parameter to SGD optimiser)
lamda=0.01                                  #Define hyperparameter
n_layers=2                                  #Define number of hidden layers in the model
n_neuron=np.logspace(0,3,4,dtype=int)       #Define number of neurons per layer
epochs=200                                   #Number of reiterations over the input data
batch_size=16                              #Number of samples per gradient update

# %%

"""Define function to return Deep Neural Network model"""

def NN_model(inputsize,n_layers,n_neuron,eta,lamda):
    model=Sequential()      
    for i in range(n_layers):       #Run loop to add hidden layers to the model
        if (i==0):                  #First layer requires input dimensions
            model.add(Dense(n_neuron,activation='relu',kernel_regularizer=regularizers.l2(lamda),input_dim=inputsize))
        else:                       #Subsequent layers are capable of automatic shape inferencing
            model.add(Dense(n_neuron,activation='relu',kernel_regularizer=regularizers.l2(lamda)))
    model.add(Dense(2,activation='softmax'))  #2 outputs - ordered and disordered (softmax for prob)
    sgd=optimizers.SGD(lr=eta)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

   
Train_accuracy=np.zeros((len(n_neuron),len(eta)))      #Define matrices to store accuracy scores as a function
Test_accuracy=np.zeros((len(n_neuron),len(eta)))       #of learning rate and number of hidden neurons for 

#for i in range(len(n_neuron)):     #run loops over hidden neurons and learning rates to calculate 
#    for j in range(len(eta)):      #accuracy scores 
#        DNN_model=NN_model(X_train.shape[1],n_layers,n_neuron[i],eta[j],lamda)
#        DNN_model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)
#        Train_accuracy[i,j]=DNN_model.evaluate(X_train,y_train)[1]
#        Test_accuracy[i,j]=DNN_model.evaluate(X_test,y_test)[1]
               
#plot_breast_data(eta,n_neuron,Train_accuracy, 'training')
#plot_breast_data(eta,n_neuron,Test_accuracy, 'testing')

# init NN
# building our neural network
n_inputs, n_features = X_train.shape
print(X_train.shape, y_train.shape)
Train_accuracy_own = np.zeros((len(n_neuron),len(eta)))      #Define matrices to store accuracy scores as a function
Test_accuracy_own  = np.zeros((len(n_neuron),len(eta)))       #of learning rate and number of hidden neurons for 
for i in range(len(n_neuron)):     #run loops over hidden neurons and learning rates to calculate 
    for j in range(len(eta)):      #accuracy scores 
        nn = Neural_Network(X_train,  y_train, costfunc = 'cross_entropy_l2reg', eta=eta[j], symbolic_differentiation = True)
        nn.add_layer(nodes = n_neuron[i], af = 'sigmoid')
        #nn.add_layer(nodes = n_neuron[i], af = 'sigmoid')
        nn.add_layer(nodes = 2, af = 'sigmoid')
        nn.feed_forward()   
        # do SGD
        # epochs, mini batches
        nn.SGD(epochs, batch_size,printout=True,**{'lambda' : lamda})
        
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
import os
plot_heatmap(eta,n_neuron,Train_accuracy_own, title = 'Train', store = True, store_dir = os.getcwd() + "/plots/prob_d/neurons_eta_heatmap_breast_")
plot_heatmap(eta,n_neuron,Test_accuracy_own, title = 'Test', store = True, store_dir = os.getcwd() + "/plots/prob_d/neurons_eta_heatmap_breast_")

