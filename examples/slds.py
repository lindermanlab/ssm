import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib
import matplotlib.pyplot as plt

import ssm
from ssm.variational import SLDSMeanFieldVariationalPosterior, SLDSTriDiagVariationalPosterior
from ssm.util import random_rotation, find_permutation

# Set the parameters of the HMM
T = 1000    # number of time bins
K = 5       # number of discrete states
D = 2       # number of latent dimensions
N = 10      # number of observed dimensions

# Make an SLDS with the true parameters
true_slds = ssm.SLDS(N, K, D, emissions="gaussian")
for k in range(K):
    true_slds.dynamics.As[k] = .95 * random_rotation(D, theta=(k+1) * np.pi/20)

# Sample training and test data from the SLDS
z, x, y = true_slds.sample(T)
z_test, x_test, y_test = true_slds.sample(T)

# Mask off some data
mask = npr.rand(T, N) < 0.75
y_masked = y * mask

# Fit an SLDS with mean field posterior
print("Fitting SLDS with SVI using structured variational posterior")
slds = ssm.SLDS(N, K, D, emissions="gaussian")
slds.initialize(y_masked, masks=mask)

q_mf = SLDSMeanFieldVariationalPosterior(slds, y_masked, masks=mask)
q_mf_elbos = slds.fit(q_mf, y_masked, masks=mask, num_iters=1000, initialize=False)
q_mf_x = q_mf.mean[0]
q_mf_y = slds.smooth(q_mf_x, y)

# Find the permutation that matches the true and inferred states
slds.permute(find_permutation(z, slds.most_likely_states(q_mf_x, y)))
q_mf_z = slds.most_likely_states(q_mf_x, y)

# Do the same with the structured posterior
print("Fitting SLDS with SVI using structured variational posterior")
slds = ssm.SLDS(N, K, D, emissions="gaussian")
slds.initialize(y_masked, masks=mask)

q_struct = SLDSTriDiagVariationalPosterior(slds, y_masked, masks=mask)
q_struct_elbos = slds.fit(q_struct, y_masked, masks=mask, num_iters=1000, initialize=False)
q_struct_x = q_struct.mean[0]
q_struct_y = slds.smooth(q_struct_x, y)

# Find the permutation that matches the true and inferred states
slds.permute(find_permutation(z, slds.most_likely_states(q_struct_x, y)))
q_struct_z = slds.most_likely_states(q_struct_x, y)

# Plot the ELBOS
plt.figure()
plt.plot(q_mf_elbos, label="MF")
plt.plot(q_struct_elbos, label="LDS")
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.legend()

# Plot the true and inferred states
plt.figure(figsize=(8,6))
xlim = (0, 1000)

plt.subplot(311)
plt.imshow(np.row_stack((z, q_mf_z, q_struct_z)), aspect="auto")
plt.yticks([0, 1, 2], ["$z_{{\\mathrm{{true}}}}$", "$z_{{\\mathrm{{mf}}}}$", "$z_{{\\mathrm{{lds}}}}$"])
plt.xlim(xlim)

plt.subplot(312)
plt.plot(x, '-k', label="True")
plt.plot(q_mf_x, '--b', label="$q_{\\text{MF}}$")
plt.plot(q_struct_x, ':r', label="$q_{\\text{LDS}}$")
plt.ylabel("$x$")
plt.xlim(xlim)

plt.subplot(313)
for n in range(N):
    plt.plot(y[:, n] + 4 * n, '-k', label="True" if n == 0 else None)
    plt.plot(q_mf_y[:, n] + 4 * n, '--b', label="MF" if n == 0 else None)
    plt.plot(q_struct_y[:, n] + 4 * n, ':r', label="LDS" if n == 0 else None)
plt.legend()
plt.ylabel("$y$")
plt.xlabel("time")
plt.xlim(xlim)

plt.show()
