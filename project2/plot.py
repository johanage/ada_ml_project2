# a script for plotting
import matplotlib.pyplot as plt
import numpy as np
import os
def plot_breast_scatter(x, y):
    plt.figure()
    plt.scatter(x[:,0],x[:,2],s=40,c=y,cmap=plt.cm.Spectral)
    plt.xlabel('Mean radius',fontweight='bold')
    plt.ylabel('Mean perimeter',fontweight='bold')
    plt.show()

    plt.figure()
    plt.scatter(x[:,5],x[:,6],s=40,c=y, cmap=plt.cm.Spectral)
    plt.xlabel('Mean compactness',fontweight='bold')
    plt.ylabel('Mean concavity',fontweight='bold')
    plt.show()


    plt.figure()
    plt.scatter(x[:,0],x[:,1],s=40,c=y,cmap=plt.cm.Spectral)
    plt.xlabel('Mean radius',fontweight='bold')
    plt.ylabel('Mean texture',fontweight='bold')
    plt.show()

    plt.figure()
    plt.scatter(x[:,2],x[:,1],s=40,c=y,cmap=plt.cm.Spectral)
    plt.xlabel('Mean perimeter',fontweight='bold')
    plt.ylabel('Mean compactness',fontweight='bold')
    plt.show()


def plot_heatmap(x,y,data,show = True, title=None, store = True, store_dir = None,
                 xlabel = None, ylabel = None, cbar_label = None, vmin=0, vmax = 1):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=vmin, vmax=vmax)

    cbar=fig.colorbar(cax)
    if cbar_label is None:
        cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    else:
        cbar.ax.set_ylabel(cbar_label,rotation=90,fontsize=fontsize)
    #cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    #cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.2f}$".format( data[j,i])
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=["1e{0:.2}".format(np.log10(i)) for i in x]
    y=["1e{0:.2}".format(np.log10(i)) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)
    if xlabel is None:
        ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
    else:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is None:
        ax.set_ylabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
    else:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
    if store:
        if store_dir is None:
            fig.savefig(os.getcwd() + "/plots/neurons_eta_heatmap_breast_" + title + ".png", dpi=150)
        else:
            fig.savefig(store_dir + title + ".png", dpi=150)
