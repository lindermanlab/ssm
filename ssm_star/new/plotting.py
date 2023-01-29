import numpy as np 

from matplotlib import pyplot as plt 

def plot_sample(x,y,z):
    
    plt.figure(figsize=(8, 9))

    plt.subplot(311)
    plt.imshow(z[None, :], aspect="auto")
    plt.yticks([0], ["$z_{{\\mathrm{{true}}}}$"])
    plt.title("(Entity-Level) Regimes")    


    plt.subplot(312)
    plt.plot(x, "-k", label="True")
    plt.ylabel("$x$")
    plt.title("States")    

    plt.subplot(313)
    N = np.shape(y)[1]  # number of observed dimensions
    spc = 1.1 * abs(y).max()
    for n in range(N):
        plt.plot(y[:, n] - spc * n, "-k", label="True" if n == 0 else None)
    plt.yticks(-spc * np.arange(N), ["$y_{}$".format(n + 1) for n in range(N)])
    plt.xlabel("time")
    plt.ylabel("$y$")
    plt.title("Observations")  

    plt.tight_layout()
    plt.show()