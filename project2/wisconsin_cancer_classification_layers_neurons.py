# to test the NN on a classification problem
from plot import plot_heatmap
import numpy as np
import matplotlib.pyplot as plt
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


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1, random_state = 0)   #Split datasets into training and testing
# scale and center
X_train = (X_train - np.mean(X_train,axis=0, keepdims=True) )/np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test,axis=0, keepdims=True) )/np.std(X_test, axis=0)
# convert to onehot vectors
y_train=to_categorical_numpy(y_train)     #Convert labels to categorical when using categorical cross entropy
y_test=to_categorical_numpy(y_test)

del temp1,temp2,temp

print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape)

eta= 1e-2                    #Define vector of learning rates (parameter to SGD optimiser)
lmbda =0.01                                  #Define hyperparameter
n_hidden_neurons = [[(1,), (1,1), (1,1,1)],
                    [(5,), (5,5), (5,5,5)],
                    [(10,), (10,10), (10,10,10)],
                    [(20,), (20,20), (20,20,20)],
                    [(30,), (30,30), (30,30,30)],
                    [(40,), (40,40), (40,40,40)],
                    [(50,), (50,50), (50,50,50)],
                    [(100,), (100,100), (100,100,100)],
                    [(500,), (500,500), (500,500,500)]]

epochs= 100                                 #Number of reiterations over the input data
batch_size=8                               #Number of samples per gradient update

# %%

"""

Define function to return Deep Neural Network model

"""
   
# init NN
# building our neural network
n_inputs, n_features = X_train.shape
print(X_train.shape, y_train.shape)
Train_accuracy_own = np.zeros((len(n_hidden_neurons),len(n_hidden_neurons[0])))      #Define matrices to store accuracy scores as a function
Test_accuracy_own  = np.zeros((len(n_hidden_neurons),len(n_hidden_neurons[0])))      #of learning rate and number of hidden neurons for 
method = 'sgd'
beta1 = 0.9
beta2 = 0.99
af = 'sigmoid'
i = 0
for net in n_hidden_neurons:
    print(net)
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    fig.suptitle("Number of layers and neurons %s with $\\lambda = 10^{%.2f}, \\eta = 10{%.2f}, m = %i$"%(str(net), np.log10(lmbda),np.log10(eta), batch_size ) )
    j = 0
    for layers in net:

        nn = Neural_Network(X_train,  y_train, costfunc = 'cross_entropy', eta=eta, w_mom = True, beta1 = beta1, beta2 = beta2, method = method, symbolic_differentiation = True)
        for neurons in layers:
            nn.add_layer(nodes = neurons, af = 'sigmoid')# weights = np.random.normal(0,.1,size = (X_train.shape[1], n_hidden_neurons) ) )        

        # output layer
        nn.add_layer(nodes = 2, af = 'sigmoid')#  weights = np.random.normal(0,1,size=(neurons, 2) )) 
        # do SGD
        print("Epochs: ", epochs, " # mini batch size :", batch_size)
        nn.SGD(epochs = epochs, size_mini_batches = batch_size, printout=True,**{'lambda' : lmbda})
        axs[0].plot(np.arange(len(nn.losses)), nn.losses, label = 'FFNN loss layers: ' + str(layers))
        axs[1].plot(np.arange(len(nn.l2norm_gradC_weights)) + 1, nn.l2norm_gradC_weights, label = 'FFNN layers: ' + str(layers))

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
        j += 1
    i += 1
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss: Cross-entropy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("$\\Vert \\nabla_{mathbm{w}} C\\Vert_2$")
    [ax.legend() for ax in axs]
    fig.savefig(os.getcwd() + "/plots/prob_d/layers_neurons_%s_cancer_FFNN.png"%(str(net)), dpi=150)
    
store_dir_train = "/plots/prob_d/%s_neurons_layers_heatmap_breast_train_"%method
store_dir_test = "/plots/prob_d/%s_neurons_layers_heatmap_breast_test_"%method

if method == 'adam':
    store_dir_train = store_dir_train + "af_%s_beta1_1e%.2f_beta2_1e%.2f"%(af,beta1, beta2)
    store_dir_test = store_dir_test + "af_%s_beta1_1e%.2f_beta2_1e%.2f"%(af,beta1, beta2)


lneurons = np.array([1, 5, 10, 20, 30, 40, 50, 100, 500])
llayers = np.array([1,2,3,])
plot_heatmap(llayers,lneurons,Train_accuracy_own, title = 'Train', type_axis = 'int',
             xlabel="Layers", ylabel="Neurons",
             store = True, store_dir = os.getcwd() + store_dir_train)
plot_heatmap(llayers,lneurons,Test_accuracy_own, title = 'Test', type_axis = 'int', 
             xlabel="Layers", ylabel="Neurons",
             store = True, store_dir = os.getcwd() + store_dir_test)
