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
K = 3     # number of discrete states
D = 1       # number of latent dimensions
N = 10      # number of observed dimensions

# Make an SLDS with the true parameters
true_slds = ssm.SLDS(N, K, D, emissions="gaussian")
for k in range(K):
    true_slds.dynamics.As[k] = .95 * random_rotation(D, theta=(k+1) * np.pi/20)

# Sample training and test data from the SLDS
z, x, y = true_slds.sample(T)
z_test, x_test, y_test = true_slds.sample(T)

# Fit an SLDS with mean field posterior
print("Fitting SLDS with SVI using structured variational posterior")
slds = ssm.SLDS(N, K, D, emissions="gaussian")
slds.initialize(y)
slds.init_state_distn.params = true_slds.init_state_distn.params

from ssm.variational import SLDSStructuredMeanFieldVariationalPosterior
q_laplace_em = SLDSStructuredMeanFieldVariationalPosterior(slds, y)
# slds.dynamics.params = true_slds.dynamics.params
# slds.emissions.params = true_slds.emissions.params
slds.fit(q_laplace_em, y, num_iters=20, method="laplace_em", initialize=False, num_samples=1)
# test Hessians
# M = (slds.M,) if isinstance(slds.M, int) else slds.M
# inputs = [np.zeros((yt.shape[0],)+M) for yt in [y]]
# masks = [np.ones_like(yt, dtype=bool) for yt in [y]]
# tags = [None] * len([y])
# input = inputs[0]
# mask = masks[0]
# tag = tags[0]
#
# # test emissions
# from autograd import hessian, grad
# obj = lambda x: np.sum(true_slds.emissions.log_likelihoods(y, input, mask, tags[0], x))
# g = grad(obj)
# hess = hessian(obj)
# hess_full = hess(x).reshape((T*D,T*D))
# hess_blocks = true_slds.emissions.hessian_log_emissions_prob(y, input, mask, tags[0], x)

# plot results
q_laplace_em_x = q_laplace_em.mean_continuous_states[0]
slds.permute(find_permutation(z, slds.most_likely_states(q_laplace_em_x, y)))
q_laplace_em_z = slds.most_likely_states(q_laplace_em_x, y)
# Ez = q_laplace_em.mean_discrete_states[0][0]
# q_laplace_em_z = np.argmax(Ez,axis=1)
# slds.permute(find_permutation(z, q_laplace_em_z ))
# Ez = q_laplace_em.mean_discrete_states[0][0]
# q_laplace_em_z = np.argmax(Ez,axis=1)
q_laplace_em_y = np.dot(q_laplace_em_x, slds.emissions.Cs[0].T) + slds.emissions.ds[0]#

# Plot the true and inferred states
plt.figure(figsize=(8,6))
xlim = (0, T)

plt.subplot(311)
plt.imshow(np.row_stack((z, q_laplace_em_z)), aspect="auto")
plt.yticks([0, 1], ["$z_{{\\mathrm{{true}}}}$", "$z_{{\\mathrm{{laplace-em}}}}$"])
plt.xlim(xlim)

plt.subplot(312)
plt.plot(x, '-k', label="True")
plt.plot(q_laplace_em_x, '--r', label="$q_{\\text{laplace-em}}$")
plt.ylabel("$x$")
plt.xlim(xlim)

plt.subplot(313)
for n in range(N):
    plt.plot(y[:, n] + 40 * n, '-k', label="True" if n == 0 else None)
    plt.plot(q_laplace_em_y[:, n] + 40 * n, '--b', label="Laplace-EM" if n == 0 else None)
plt.legend()
plt.ylabel("$y$")
plt.xlabel("time")
plt.xlim(xlim)

# compare with mean field
# slds_mf = ssm.SLDS(N, K, D, emissions="gaussian")
# slds_mf.initialize(y)
# slds_mf.init_state_distn.params = true_slds.init_state_distn.params
# q_mf = SLDSMeanFieldVariationalPosterior(slds_mf, y)
# q_mf_elbos = slds_mf.fit(q_mf, y, num_iters=1000, initialize=False)
# q_mf_x = q_mf.mean[0]
# q_mf_y = slds_mf.smooth(q_mf_x, y)
