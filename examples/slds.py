import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib
import matplotlib.pyplot as plt

from ssm.models import GaussianSLDS
from ssm.util import random_rotation

# Set the parameters of the HMM
T = 10000   # number of time bins
K = 5       # number of discrete states
D = 2       # number of latent dimensions
N = 10      # number of observed dimensions

# Make an HMM with the true parameters
true_slds = GaussianSLDS(N, K, D)
for k in range(K):
	true_slds.As[k] = .95 * random_rotation(D, theta=(k+1) * np.pi/20)
z, x, y = true_slds.sample(T)
z_test, x_test, y_test = true_slds.sample(T)

# Fit models
N_sgd_iters = 5000

print("Fitting HMM with SGD")
slds = GaussianSLDS(N, K, D)
slds_elbos, variational_params = slds.fit(y, num_iters=N_sgd_iters, step_size=.001)
slds_x = variational_params[0][0]
slds_Ez, _ = slds.expected_states(variational_params[0])
slds_z = np.argmax(slds_Ez, axis=1)
slds_y = slds.smooth(variational_params[0])

plt.figure(figsize=(8,6))
plt.subplot(311)
plt.imshow(np.column_stack((z, slds_z)).T, aspect="auto")
# plt.ylabel("$z$")
plt.yticks([0, 1], ["$z_{{\\mathrm{{true}}}}$", "$z_{{\\mathrm{{inf}}}}$"])
plt.xlim(0, T)
plt.subplot(312)
plt.plot(x, '-k')
plt.plot(slds_x, ':')
plt.ylabel("$x$")
plt.xlim(0, T)
plt.subplot(313)
plt.plot(y + 4 * np.arange(N), '-k')
plt.plot(slds_y + 4 * np.arange(N), ':')
plt.ylabel("$y$")
plt.xlabel("time")
plt.xlim(0, T)
plt.show()
