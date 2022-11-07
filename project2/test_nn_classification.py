# to test the NN on a classification problem
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import sys
# for reproducability
np.random.seed(0)

# load mnist dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target


# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))

# choose some random images to display
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)

#for i, image in enumerate(digits.images[random_indices]):
#    plt.subplot(1, 5, i+1)
#    plt.axis('off')
#    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#    plt.title("Label: %d" % digits.target[random_indices[i]])
#    plt.show()

from sklearn.model_selection import train_test_split

# one-liner from scikit-learn library
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                    test_size=test_size)

print("Number of training images: " + str(len(X_train)))
print("Number of test images: " + str(len(X_test)))

from neural_network import Neural_Network
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
# init NN
nn = Neural_Network(X_train,  Y_train[:,np.newaxis], costfunc = 'cross_entropy', eta=1e-3)

# building our neural network

n_inputs, n_features = X_train.shape
n_hidden_neurons = 50
n_categories = 10

# we make the weights normally distributed using numpy.random.randn
# weights and bias in the hidden layer
hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros(n_categories) + 0.01

nn.add_layer(nodes = n_hidden_neurons, af = 'sigmoid', weights = hidden_weights, bias = hidden_bias[:,np.newaxis])
nn.add_layer(nodes = n_categories,     af = 'softmax', weights = output_weights, bias = output_bias[:,np.newaxis])
nn.feed_forward()


# do SGD
# epochs, mini batches
#nn.SGD(200, 100)
# set data to original and predict using weights computed with SGD
#nn.target = nn.Ydata_full
#nn.a[0] = nn.Xdata_full
#nn.feed_forward()

