# small script to plot activation functions
import numpy as np
import matplotlib.pyplot as plt
from plot import *
import os
x = np.linspace(-5,5,100)
fig, ax = plt.subplots()
fig.suptitle("Activation functions")
for af in ['sigmoid', 'relu', 'leaky_relu', 'tanh']:
    fig, ax = plot_activation_functions(af, x, fig, ax)
ax.legend()
ax.grid()
ax.set_ylim(-1.1,1.1)
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.savefig(os.getcwd() + "/plots/prob_c/activation_functions_xin_-1_1.png", dpi=150)
