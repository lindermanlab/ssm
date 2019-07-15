import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "dark brown"]
colors = sns.xkcd_palette(color_names)

from ssm import LDS
from ssm.util import random_rotation

# Set the parameters of the HMM
T = 1000   # number of time bins
D = 2      # number of latent dimensions
N = 10     # number of observed dimensions

# Make an LDS with somewhat interesting dynamics parameters
true_lds = LDS(N, D, emissions="gaussian")
A0 = .99 * random_rotation(D, theta=np.pi/20)
# S = (1 + 3 * npr.rand(D))
S = np.arange(1, D+1)
R = np.linalg.svd(npr.randn(D, D))[0] * S
A = R.dot(A0).dot(np.linalg.inv(R))
b = npr.randn(D)
true_lds.dynamics.As[0] = A
true_lds.dynamics.bs[0] = b

_, x, y = true_lds.sample(T)

print("Fitting LDS with SVI")

# Create the model and initialize its parameters
lds = LDS(N, D, emissions="gaussian")
lds.initialize(y)
q_mf_elbos, q_mf = lds.fit(y, method="bbvi", variational_posterior="mf", num_iters=5000, stepsize=0.1, initialize=False)
# Get the posterior mean of the continuous states
q_mf_x = q_mf.mean[0]
# Smooth the data under the variational posterior
q_mf_y = lds.smooth(q_mf_x, y)


print("Fitting LDS with SVI using structured variational posterior")
lds = LDS(N, D, emissions="gaussian")
lds.initialize(y)
q_struct_elbos, q_struct = lds.fit(y, method="bbvi", variational_posterior="lds", num_iters=2000, stepsize=0.1, initialize=False)
# Get the posterior mean of the continuous states
q_struct_x = q_struct.mean[0]
# Smooth the data under the variational posterior
q_struct_y = lds.smooth(q_struct_x, y)

print("Fitting LDS with Laplace EM")
lds = LDS(N, D, emissions="gaussian")
lds.initialize(y)
q_lem_elbos, q_lem = lds.fit(y, method="laplace_em", variational_posterior="structured_meanfield",
                             num_iters=10, initialize=False)
# Get the posterior mean of the continuous states
q_lem_x = q_lem.mean_continuous_states[0]
# Smooth the data under the variational posterior
q_lem_y = lds.smooth(q_lem_x, y)

# Plot the ELBOs
plt.figure()
plt.plot(q_mf_elbos, label="MF")
plt.plot(q_struct_elbos, label="LDS")
plt.plot(q_lem_elbos, label="Laplace-EM")
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.legend()

plt.figure(figsize=(8,4))
plt.plot(x + 4 * np.arange(D), '-k')
for d in range(D):
    plt.plot(q_mf_x[:,d] + 4 * d, '-', color=colors[0], label="MF" if d==0 else None)
    plt.plot(q_struct_x[:,d] + 4 * d, ':', color=colors[1], label="Struct" if d==0 else None)
    plt.plot(q_lem_x[:,d] + 4 * d, '--', color=colors[2], label="Laplace-EM" if d==0 else None)
plt.ylabel("$x$")
plt.xlim((0,200))
plt.legend()

# Plot the smoothed observations
plt.figure(figsize=(8,4))
for n in range(N):
    plt.plot(y[:, n] + 4 * n, '-k', label="True" if n == 0 else None)
    plt.plot(q_mf_y[:, n] + 4 * n, '--', color=colors[0], label="MF" if n == 0 else None)
    plt.plot(q_struct_y[:, n] + 4 * n, ':',  color=colors[1], label="Struct" if n == 0 else None)
    plt.plot(q_lem_y[:, n] + 4 * n, '--',  color=colors[2], label="Laplace-EM" if n == 0 else None)
plt.legend()
plt.xlabel("time")
plt.xlim((0,200))
