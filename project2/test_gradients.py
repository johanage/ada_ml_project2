# to check if the gradients in the NN is implemented correctly

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from neural_network import Neural_Network
test_size = 0.2
nsamples = 5
x = np.linspace(0,1,nsamples)
X = x[:,np.newaxis]
y = X + X**2
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = test_size, random_state=3155)
costfunc = 'ols'

# setting up a single layer perceptron
#generating the bias and weights
n_features = X.shape[1]
hidden_nodes = 2
hidden_activation_function = 'sigmoid'
hidden_bias =    np.random.normal(0, 0.1, size=(hidden_nodes, 1))
hidden_weights = np.random.normal(0, 0.1, size=(n_features, hidden_nodes) )
output_bias =    np.random.normal(0,0.1, size=(nsamples, 1))
output_weights = np.random.normal(0,0.1, size=(hidden_nodes, nsamples) )

# init NN
nn1 = Neural_Network(X_train, Y_train, costfunc, eta=1e-2)
nn2 = Neural_Network(X_train, Y_train, costfunc, eta=1e-2, specify_grad = True)
for nn in [nn1, nn2]:
    nn.add_layer(hidden_nodes, hidden_activation_function, hidden_weights, hidden_bias)
    nn.add_layer(nsamples, 'linear', output_weights, output_bias)
    nn.feed_forward()
    nn.backpropagation()

print("Result after feed forward is equal:", nn1.a[nn1.layers], nn2.a[nn.layers])

print("Computing gradients using autograd:\n")
grad1 = nn1.nabla_a_C(nn1.target, nn1.a[nn1.layers])
print(grad1)
print("Computing gradients using symbolic differentiation:\n")
grad2 = nn2.nabla_a_C(nn2.target, nn2.a[nn2.layers])
print(grad2)

print("Computing delta_L using autograd differentiation:\n")
print(nn1.delta[nn1.layers] )
print("Computing delta_L using symbolic differentiation:\n")
print(nn2.delta[nn2.layers] )


print("Computing sigma_prime( zL ) using autograd differentiation:\n")
print(nn1.sigma_prime('sigmoid', nn1.z[nn1.layers] ) )
print("Computing sigma_prime( zL ) using symbolic differentiation:\n")
print(nn2.sigma_prime('sigmoid', nn2.z[nn2.layers] ) )
