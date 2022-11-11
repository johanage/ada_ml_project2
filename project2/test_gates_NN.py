# simple classification test and comparison with sklearn MLP
# the gates: XOR, OR, AND
# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from neural_network import Neural_Network
# ensure the same random numbers appear every time
np.random.seed(0)

# Design matrix
X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)

# The XOR gate
yXOR = np.array( [ 0, 1 ,1, 0])
# The OR gate
yOR = np.array( [ 0, 1 ,1, 1])
# The AND gate
yAND = np.array( [ 0, 0 ,0, 1])

# Defining the neural network
n_inputs, n_features = X.shape
n_hidden_neurons = 2
n_categories = 2
n_features = 2

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)

# store models for later use
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
DNN_own = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
epochs = 100
size_mini_batch = yXOR.shape[0]
print("epochs", epochs, "size of the mini batches", size_mini_batch)
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
        dnn.fit(X, yXOR)
        DNN_scikit[i][j] = dnn
        dnn_own = Neural_Network(X,  yXOR[:,np.newaxis], costfunc = 'ridge', eta=eta)
        dnn_own.add_layer(nodes = n_hidden_neurons, af = 'sigmoid')
        dnn_own.add_layer(nodes = n_categories, af = 'sigmoid')
        dnn_own.SGD(epochs, size_mini_batch, **{'lambda' : lmbd})
        dnn_own.eval_score("classification")
        DNN_own[i][j] = dnn_own
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on data set: ", dnn.score(X, yXOR), " and own implementation: ", dnn_own.score)
        print()

sns.set()
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy_own = np.zeros((len(eta_vals), len(lmbd_vals)))
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_scikit[i][j]
        dnn_own = DNN_own[i][j]
        dnn_own.a[0] = X
        dnn_own.feed_forward()
        dnn_own.eval_score("classification")
        test_pred = dnn.predict(X)
        test_accuracy[i][j] = accuracy_score(yXOR, test_pred)
        test_accuracy_own[i][j] = dnn_own.score
        
fig, axs = plt.subplots(1, 2, figsize = (10, 5))
sns.heatmap(test_accuracy, annot=True, ax=axs[0], cmap="viridis")
sns.heatmap(test_accuracy_own, annot=True, ax=axs[1], cmap="viridis")
axs[0].set_title("Test Accuracy sklearn")
axs[1].set_title("Test Accuracy own")
[ax.set_ylabel("$\eta$") for ax in axs]
[ax.set_xlabel("$\lambda$") for ax in axs]
plt.show()
