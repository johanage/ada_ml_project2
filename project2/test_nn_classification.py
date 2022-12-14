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

plot = False
if plot:
    for i, image in enumerate(digits.images[random_indices]):
        plt.subplot(1, 5, i+1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Label: %d" % digits.target[random_indices[i]])
        plt.show()

from sklearn.model_selection import train_test_split

# one-liner from scikit-learn library
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                    test_size=test_size)
print("Number of training images: " + str(len(X_train)))
print("Number of test images: " + str(len(X_test)))

from neural_network import Neural_Network, probs_to_binary, accuracy, to_categorical_numpy
from test import test_func_poly_deg_p
import numpy as np
import matplotlib.pyplot as plt
# init NN
# building our neural network
n_inputs, n_features = X_train.shape
n_hidden_neurons = 50
n_categories = 10

#Y_train_onehot, Y_test_onehot = to_categorical(Y_train), to_categorical(Y_test)
Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)
print("input training target: ", Y_train_onehot)

# init and add layers to the NN
nn = Neural_Network(X_train,  Y_train_onehot, costfunc = 'cross_entropy', eta=1e-2, w_mom = True, method = 'sgd', symbolic_differentiation = False)
nn.add_layer(nodes = n_hidden_neurons, af = 'sigmoid')
nn.add_layer(nodes = n_categories,     af = 'softmax')
nn.feed_forward()

# do SGD
# epochs, mini batches
nn.SGD(100, 16,  **{'lambda' : 1e-3}, plot=True, store_grads = False, store_activation_output = False)
# set data to original and predict using weights computed with SGD
aL_test = nn.predict(X_test)
p2b = probs_to_binary(probabilities = aL_test)
acc = accuracy(y = Y_test_onehot, a = p2b)
print("Accuracy: ", acc)

print(aL_test[:10])
print(Y_test[:10])
