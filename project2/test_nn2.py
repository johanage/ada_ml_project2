# test cript for the NN
from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(3155)
X_train = np.linspace(-1, 1,1000)[:,np.newaxis]
Y_train = test_func_poly_deg_p(deg = 2, avec = [0,1,1], x = X_train)
Y_train_noisy = Y_train + np.random.normal(0,.1,size=Y_train.shape)
# building our neural network

n_inputs, n_features = X_train.shape
n_hidden_neurons = 10
n_categories = np.copy(n_inputs)
# we make the weights normally distributed using numpy.random.randn
# weights and bias in the hidden layer
hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros((n_hidden_neurons, 1)) + 0.01

# weights and bias in the output layer
output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros((n_categories, 1)) + 0.01


#x = (x - np.mean(x))/np.std(x) # pre-processing of input
eta = 0.01
nn = Neural_Network(X_train, Y_train, costfunc = 'ols', eta = eta)
nn.add_layer(nodes = n_hidden_neurons, af = 'sigmoid', weights = hidden_weights, bias = hidden_bias)
nn.output_layer(af = 'linear', weights = output_weights, bias = output_bias)
nn.feed_forward()
nn.backpropagation()
for iter in range(1):
    nn.backpropagation()
    nn.update_weights(200)
    nn.feed_forward()
    print(np.mean(np.square(nn.target-nn.a[nn.layers])))

# setup the feed-forward pass, subscript h = hidden layer

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def feed_forward_train(X):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias.T
    # activation in the hidden layer
    a_h = sigmoid(z_h)
    
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias.T
    
    # for backpropagation need activations in hidden and output layers
    return a_h, z_o

def feed_forward_out(X):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias.T
    # activation in the hidden layer
    a_h = z_h
    
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias.T
    
    # for backpropagation need activations in hidden and output layers
    return a_h, z_o


def backpropagation(X, Y):
    a_h, z_o = feed_forward_train(X)

    # error in the output layer
    error_output = z_o - Y
    # error in the hidden layer
    error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)

    # gradients for the output layer
    output_weights_gradient = np.matmul(a_h.T, error_output)
    output_bias_gradient = np.sum(error_output, axis=0)

    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(X.T, error_hidden)
    hidden_bias_gradient = np.sum(error_hidden, axis=0)

    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

for i in range(1):
    # calculate gradients
    dWo, dBo, dWh, dBh = backpropagation(X_train, Y_train)

    # regularization term gradients
    dWo +=  output_weights
    dWh +=  hidden_weights

    # update weights and biases
    output_weights -= eta * dWo
    output_bias -=    eta * np.sum(dBo, axis=0, keepdims=True)
    hidden_weights -= eta * dWh
    hidden_bias -=    eta * np.sum(dBh, axis=0, keepdims=True)

a_L, z_L = feed_forward_out(X_train)

print("mortens implementation output vs my output")
print(" Mortens : ", z_L)
print(" Own : ", nn.a[nn.layers])


#plt.plot(x, y, label='y')
#plt.plot(x, nn.a[nn.layers], label='$a^L$ NN')
#plt.plot(x, nn.z[nn.layers], label='$z^L$ NN')
#plt.legend()
#plt.show()
