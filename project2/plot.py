# a script for plotting
import matplotlib.pyplot as plt
import numpy as np

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


def plot_breast_data(x,y,data,title=None):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)

    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)

    ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    plt.show()

