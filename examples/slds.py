import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib
import matplotlib.pyplot as plt

from ssm.models import SLDS, LDS
from ssm.util import random_rotation, find_permutation

# Set the parameters of the HMM
T = 1000    # number of time bins
K = 5       # number of discrete states
D = 2       # number of latent dimensions
N = 10      # number of observed dimensions

# Make an SLDS with the true parameters
true_slds = SLDS(N, K, D, observations="gaussian")
for k in range(K):
	true_slds.As[k] = .95 * random_rotation(D, theta=(k+1) * np.pi/20)
z, x, y = true_slds.sample(T)
z_test, x_test, y_test = true_slds.sample(T)

# Mask off some data
mask = npr.rand(T, N) < 0.5

# print("Fitting LDS with SVI")
# lds = LDS(N, D, observations="gaussian")
# lds_elbos, (lds_x, lds_x_var) = lds.fit(y * mask, masks=mask, num_iters=100)

print("Fitting SLDS with SVI")
slds = SLDS(N, K, D, observations="gaussian")
slds_elbos, (slds_x, slds_x_var) = slds.fit(y * mask, masks=mask, num_iters=1000, print_intvl=10)

# Find the permutation that matches the true and inferred states
slds.permute(find_permutation(z, slds.most_likely_states(slds_x)))
slds_z = slds.most_likely_states(slds_x)

# Smooth the observations
slds_y = slds.smooth(slds_x)

# Plot the true and inferred states
plt.figure(figsize=(8,6))
xlim = (0, 200)

plt.subplot(311)
plt.imshow(np.column_stack((z, slds_z)).T, aspect="auto")
plt.yticks([0, 1], ["$z_{{\\mathrm{{true}}}}$", "$z_{{\\mathrm{{inf}}}}$"])
plt.xlim(xlim)

plt.subplot(312)
plt.plot(x, '-k')
plt.plot(slds_x, ':')
plt.ylabel("$x$")
plt.xlim(xlim)

plt.subplot(313)
plt.plot(y + 4 * np.arange(N), '-k')
# plt.plot(slds_y + 4 * np.arange(N), ':')
plt.ylabel("$y$")
plt.xlabel("time")
plt.xlim(xlim)

plt.show()
