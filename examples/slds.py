import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib
import matplotlib.pyplot as plt

import ssm
from ssm.variational import SLDSMeanFieldVariationalPosterior, SLDSTriDiagVariationalPosterior, \
	SLDSStructuredMeanFieldVariationalPosterior
from ssm.util import random_rotation, find_permutation

# Set the parameters of the HMM
T = 10000   # number of time bins
<<<<<<< HEAD
K = 5      # number of discrete states
=======
K = 5       # number of discrete states
>>>>>>> 2d9b6634affe20a6cfab733a65d7e337915f8257
D = 2       # number of latent dimensions
N = 50      # number of observed dimensions

# Make an SLDS with the true parameters
true_slds = ssm.SLDS(N, K, D, emissions="bernoulli_orthog")
for k in range(K):
    true_slds.dynamics.As[k] = .95 * random_rotation(D, theta=(k+1) * np.pi/20)
    # true_slds.dynamics.bs[k] = .1 * npr.randn(D)
# true_slds.dynamics.Sigmas = np.tile(0.1 * np.eye(D)[None, :, :], (K, 1, 1))

# Sample training and test data from the SLDS
z, x, y = true_slds.sample(T)
z_test, x_test, y_test = true_slds.sample(T)

# Fit an SLDS with mean field posterior
print("Fitting SLDS with SVI using structured variational posterior")
slds = ssm.SLDS(N, K, D, emissions="bernoulli")
slds.initialize(y)

# q_mf = SLDSMeanFieldVariationalPosterior(slds, y_masked, masks=mask)
# q_mf_elbos = slds.fit(q_mf, y_masked, masks=mask, num_iters=1000, initialize=False)
# q_mf_x = q_mf.mean[0]
# q_mf_y = slds.smooth(q_mf_x, y)

# # Find the permutation that matches the true and inferred states
# slds.permute(find_permutation(z, slds.most_likely_states(q_mf_x, y)))
# q_mf_z = slds.most_likely_states(q_mf_x, y)

# # Do the same with the structured posterior
# print("Fitting SLDS with SVI using structured variational posterior")
# slds = ssm.SLDS(N, K, D, emissions="gaussian")
# slds.initialize(y_masked, masks=mask)

# q_struct = SLDSTriDiagVariationalPosterior(slds, y_masked, masks=mask)
# q_struct_elbos = slds.fit(q_struct, y_masked, masks=mask, num_iters=1000, initialize=False)
# q_struct_x = q_struct.mean[0]
# q_struct_y = slds.smooth(q_struct_x, y)

# # Find the permutation that matches the true and inferred states
# slds.permute(find_permutation(z, slds.most_likely_states(q_struct_x, y)))
# q_struct_z = slds.most_likely_states(q_struct_x, y)

q_laplace_em = SLDSStructuredMeanFieldVariationalPosterior(slds, y)
<<<<<<< HEAD
q_lem_elbos = slds.fit(q_laplace_em, y, method="laplace_em", num_iters=10, initialize=False, num_samples=1)
=======
q_lem_elbos = slds.fit(q_laplace_em, y, method="laplace_em", num_iters=100, initialize=False)
>>>>>>> 2d9b6634affe20a6cfab733a65d7e337915f8257
q_lem_Ez, q_lem_x = q_laplace_em.mean[0]
q_lem_y = slds.smooth(q_lem_x, y)

# Find the permutation that matches the true and inferred states
slds.permute(find_permutation(z, slds.most_likely_states(q_lem_x, y)))
q_lem_z = slds.most_likely_states(q_lem_x, y)

# Linearly transform the x's to match the true x's
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(q_lem_x, x)
q_lem_x_trans = lr.predict(q_lem_x)

# Plot the ELBOS
plt.figure()
plt.plot(q_lem_elbos, label="MF")
# plt.plot(q_struct_elbos, label="LDS")
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.legend()
plt.tight_layout()

# Plot the true and inferred states
plt.figure(figsize=(8,9))
xlim = (0, 1000)

plt.subplot(411)
plt.imshow(z[None, :], aspect="auto")
# plt.imshow(np.row_stack((z, q_mf_z, q_struct_z)), aspect="auto")
plt.imshow(np.row_stack((z, q_lem_z)), aspect="auto")
# plt.yticks([0, 1, 2], ["$z_{{\\mathrm{{true}}}}$", "$z_{{\\mathrm{{mf}}}}$", "$z_{{\\mathrm{{lds}}}}$"])
plt.yticks([0, 1, 2], ["$z_{{\\mathrm{{true}}}}$", "$z_{{\\mathrm{{L. EM}}}}$"])
plt.xlim(xlim)
plt.title("True and Most Likely Inferred States")

plt.subplot(412)
plt.imshow(q_lem_Ez[0].T, aspect="auto", cmap="Greys")
plt.xlim(xlim)
plt.title("Inferred State Probability")

plt.subplot(413)
plt.plot(x, '-k', label="True")
# plt.plot(q_mf_x, '--b', label="$q_{\\text{MF}}$")
# plt.plot(q_struct_x, ':r', label="$q_{\\text{LDS}}$")
plt.plot(q_lem_x_trans, ':r', label="$q_{\\text{Laplace}}$")
plt.ylabel("$x$")
plt.xlim(xlim)

plt.subplot(414)
spc = 1.1 * abs(y).max()
for n in range(N):
    plt.plot(y[:, n] - spc * n, '-k', label="True" if n == 0 else None)
    # plt.plot(q_mf_y[:, n] - spc * n, '--b', label="MF" if n == 0 else None)
    # plt.plot(q_struct_y[:, n] - spc * n, ':r', label="LDS" if n == 0 else None)
    plt.plot(q_lem_y[:, n] - spc * n, '--b', label="Laplace EM" if n == 0 else None)
plt.legend()
plt.ylabel("$y$")
plt.yticks(-spc * np.arange(N), ["$y_{}$".format(n+1) for n in range(N)])
plt.xlabel("time")
plt.xlim(xlim)

plt.show()
