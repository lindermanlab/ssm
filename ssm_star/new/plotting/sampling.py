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
from lds.types import NumpyArray2D


color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")


def plot_sample(x: NumpyArray2D, y: NumpyArray2D, z: typing.List[int]) -> None:
    """
    Arguments:
        x: continuous states
        y: observations
        z: (discrete) regimes, entity-level
    """

    plt.figure(figsize=(8, 9))

    plt.subplot(311)
    plt.imshow(np.array(z)[None, :], aspect="auto")
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
