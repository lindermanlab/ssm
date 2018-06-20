import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib
import matplotlib.pyplot as plt

from ssm.stacked_slds import StackedSLDS
from ssm.util import random_rotation, find_permutation

# Set the parameters of the HMM
T = 1000    # number of time bins
K = 5      # number of discrete states
D = 2       # number of latent dimensions
N = 10      # number of observed dimensions
L = 2       # number of layers

# Make an SLDS with the true parameters
true_slds = StackedSLDS(L, N, K, D)
for l in range(L):
    for k in range(K):
        true_slds.list_of_dynamics[l].As[k] = .95 * random_rotation(D, theta=(k+1) * np.pi/20)

Ps = .999 * np.eye(K) + .001 * npr.rand(K, K)
Ps /= Ps.sum(axis=1, keepdims=True)
true_slds.list_of_transitions[0].log_Ps = np.log(Ps)

z, x, y = true_slds.sample(T)
z_test, x_test, y_test = true_slds.sample(T)

print("Fitting SSLDS with SVI")
slds = StackedSLDS(L, N, K, D)
slds_elbos, (slds_x, slds_x_var) = slds.fit(y, num_iters=1000, print_intvl=10)

plt.figure(figsize=(12, 8))
for l in range(L):
    plt.subplot(2*L+1, 1, 2*l+1)
    plt.imshow(z[l][None, :], aspect="auto")
    plt.ylabel("$z_{{ {0} }}$".format(l+1))
    plt.xlim(0, T)
    plt.yticks([])
    plt.xticks([])

    plt.subplot(2*L+1, 1, 2*l+2)
    plt.plot(x[l])
    plt.plot(slds_x[l])
    plt.ylabel("$x_{{ {0} }}$".format(l+1))
    plt.xlim(0, T)
    plt.yticks([])
    plt.xticks([])

plt.subplot(2*L+1, 1, 2*L+1)
plt.plot(y + 8 * np.arange(N))
plt.ylabel("$y$")
plt.xlabel("time")
plt.xlim(0, T)
plt.yticks([])

plt.show()

# # Find the permutation that matches the true and inferred states
# slds.permute(find_permutation(z, slds.most_likely_states(slds_x, y)))
# slds_z = slds.most_likely_states(slds_x, y)

# # Smooth the observations
# slds_y = slds.smooth(slds_x, y)

# # Plot the true and inferred states
# plt.figure(figsize=(8,6))
# xlim = (0, 200)

# plt.subplot(311)
# plt.imshow(np.column_stack((z, slds_z)).T, aspect="auto")
# plt.yticks([0, 1], ["$z_{{\\mathrm{{true}}}}$", "$z_{{\\mathrm{{inf}}}}$"])
# plt.xlim(xlim)

# plt.subplot(312)
# plt.plot(x, '-k')
# plt.plot(slds_x, ':')
# plt.ylabel("$x$")
# plt.xlim(xlim)

# plt.subplot(313)
# plt.plot(y + 4 * np.arange(N), '-k')
# # plt.plot(slds_y + 4 * np.arange(N), ':')
# plt.ylabel("$y$")
# plt.xlabel("time")
# plt.xlim(xlim)

# plt.show()
