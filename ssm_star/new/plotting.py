"""
We run SLDS from the Linderman `ssm` package
(located at https://github.com/lindermanlab/ssm/blob/master/examples/slds.py)
on our own generated data in order to see how well we can segment.
"""
import typing

import numpy as np
import numpy.random as npr


npr.seed(0)

import matplotlib.pyplot as plt
import seaborn as sns
from ssm_star.lds import SLDS

from lds.piecewise.metrics import compute_regime_labeling_accuracy
from lds.types import NumpyArray1D, NumpyArray2D


color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

def plot_sample(x : NumpyArray2D, y : NumpyArray2D, z : typing.List[int]) -> None:
    """
    Arguments:
        x: continuous states
        y: observations
        z: (discrete) regimes, entity-level
    """
    
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

def plot_elbos(q_elbos: NumpyArray1D) -> None:
    plt.figure()
    plt.plot(q_elbos)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.tight_layout()
    plt.show()

def plot_results_for_one_entity(
    q, 
    slds: SLDS, 
    y: NumpyArray2D,
    x_true: NumpyArray2D,
    z_true: typing.List[int],
    system_input: NumpyArray2D,
) -> None:

    # TODO: Make `method` and `variational_posterior` arguments to the function
    # TODO: Relabel q_x, q_y, q_z as expectations.
    # TODO: Figure out what all the entries of q_Ez are.
    q_Ez, q_x = q.mean[0]
    q_y = slds.smooth(q_x, y, system_input = system_input)
    z_most_likely = slds.most_likely_states(q_x, y, system_input = system_input)
    pct_correct_regimes = compute_regime_labeling_accuracy(z_most_likely, z_true)
    print(f"\n Pct correct segmentations: {pct_correct_regimes:.02f}.")

    # Linearly transform the x's to match the true x's
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(q_x, x_true)
    q_x_trans = lr.predict(q_x)

    # Plot the true and inferred states
    plt.figure(figsize=(8, 9))

    plt.subplot(411)
    plt.imshow(np.array(z_true)[None, :], aspect="auto")
    plt.imshow(np.row_stack((z_true, z_most_likely)), aspect="auto")
    plt.axhline(y=0.5, color='r', linestyle='-')
    plt.yticks([0, 1], ["$z_{{\\mathrm{{true}}}}$", "$z_{{\\mathrm{{est}}}}$"])
    plt.title("True and Most Likely Inferred States")

    plt.subplot(412)
    plt.imshow(q_Ez[0].T, aspect="auto", cmap="Greys")
    plt.title("Inferred State Probability")

    plt.subplot(413)
    plt.plot(x_true, "-k", label="True")
    plt.plot(q_x_trans, ":r", label="$q_{\\text{Laplace}}$")
    plt.ylabel("$x$")

    plt.subplot(414)
    N = np.shape(y)[1]  # number of observed dimensions
    spc = 1.1 * abs(y).max()
    for n in range(N):
        plt.plot(y[:, n] - spc * n, "-k", label="True" if n == 0 else None)
        plt.plot(q_y[:, n] - spc * n, "--b", label="Laplace EM" if n == 0 else None)
    plt.yticks(-spc * np.arange(N), ["$y_{}$".format(n + 1) for n in range(N)])
    plt.xlabel("time")
    plt.ylabel("$y$")

    plt.tight_layout()
    plt.show()
