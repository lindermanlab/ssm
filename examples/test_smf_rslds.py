import os
import pickle

import autograd.numpy as np
import autograd.numpy.random as npr
# npr.seed(12345)
npr.seed(0)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import ssm
from ssm.variational import SLDSMeanFieldVariationalPosterior, SLDSTriDiagVariationalPosterior
from ssm.util import random_rotation

# Global parameters
T = 10000
K = 4
D_obs = 10
D_latent = 2

# Helper functions for plotting results
def plot_trajectory(z, x, ax=None, ls="-"):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=colors[z[start] % len(colors)],
                alpha=1.0)

    return ax


def plot_most_likely_dynamics(model,
    xlim=(-4, 4), ylim=(-3, 3), nxpts=30, nypts=30,
    alpha=0.8, ax=None, figsize=(3, 3)):

    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    log_Ps = model.transitions.log_transition_matrices(
        xy, np.zeros((nxpts * nypts, 0)), np.ones_like(xy, dtype=bool), None)
    z = np.argmax(log_Ps[:, 0, :], axis=-1)
    z = np.concatenate([[z[0]], z])

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax

# Simulate the nascar data
def make_nascar_model():
    As = [random_rotation(D_latent, np.pi/24.),
      random_rotation(D_latent, np.pi/48.)]

    # Set the center points for each system
    centers = [np.array([+2.0, 0.]),
           np.array([-2.0, 0.])]
    bs = [-(A - np.eye(D_latent)).dot(center) for A, center in zip(As, centers)]

    # Add a "right" state
    As.append(np.eye(D_latent))
    bs.append(np.array([+0.1, 0.]))

    # Add a "right" state
    As.append(np.eye(D_latent))
    bs.append(np.array([-0.25, 0.]))

    # Construct multinomial regression to divvy up the space
    w1, b1 = np.array([+1.0, 0.0]), np.array([-2.0])   # x + b > 0 -> x > -b
    w2, b2 = np.array([-1.0, 0.0]), np.array([-2.0])   # -x + b > 0 -> x < b
    w3, b3 = np.array([0.0, +1.0]), np.array([0.0])    # y > 0
    w4, b4 = np.array([0.0, -1.0]), np.array([0.0])    # y < 0
    Rs = np.row_stack((100*w1, 100*w2, 10*w3,10*w4))
    r = np.concatenate((100*b1, 100*b2, 10*b3, 10*b4))

    true_rslds = ssm.SLDS(D_obs, K, D_latent,
                      transitions="recurrent_only",
                      dynamics="diagonal_gaussian",
                      emissions="gaussian_orthog",
                      single_subspace=True)
    true_rslds.dynamics.mu_init = np.tile(np.array([[0, 1]]), (K, 1))
    true_rslds.dynamics.sigmasq_init = 1e-4 * np.ones((K, D_latent))
    true_rslds.dynamics.As = np.array(As)
    true_rslds.dynamics.bs = np.array(bs)
    true_rslds.dynamics.sigmasq = 1e-4 * np.ones((K, D_latent))

    true_rslds.transitions.Rs = Rs
    true_rslds.transitions.r = r

    true_rslds.emissions.inv_etas = np.log(1e-2) * np.ones((1, D_obs))
    return true_rslds

# Sample from the model
true_rslds = make_nascar_model()
z, x, y = true_rslds.sample(T=T)

# Fit a robust rSLDS with its default initialization
rslds = ssm.SLDS(D_obs, K, D_latent,
             transitions="recurrent_only",
             dynamics="gaussian",
             emissions="gaussian_orthog",
             single_subspace=True)

# Initialize the model with the observed data.  It is important
# to call this before constructing the variational posterior since
# the posterior constructor initialization looks at the rSLDS parameters.
rslds.initialize(y)
rslds.init_state_distn.params = true_rslds.init_state_distn.params

from ssm.variational import SLDSStructuredMeanFieldVariationalPosterior
q_laplace_em = SLDSStructuredMeanFieldVariationalPosterior(rslds, y)
elbos = rslds.fit(q_laplace_em, y, num_iters=100, method="laplace_em", initialize=False, num_samples=1)
xhat = q_laplace_em.mean_continuous_states[0]
zhat = rslds.most_likely_states(xhat, y)
yhat = np.dot(xhat, rslds.emissions.Cs[0].T) + rslds.emissions.ds[0]#

# Uncomment this to fit with stochastic variational inference instead
# rslds_svi = ssm.SLDS(D_obs, K, D_latent,
#              transitions="recurrent_only",
#              dynamics="gaussian",
#              emissions="gaussian_orthog",
#              single_subspace=True)
# rslds_svi.initialize(y)
# rslds_svi.init_state_distn.params = true_rslds.init_state_distn.params
# q_svi = SLDSTriDiagVariationalPosterior(rslds_svi, y)
# elbos = rslds_svi.fit(q_svi, y, method="svi", num_iters=1000, initialize=False)
# xhat_svi = q_svi.mean[0]
# zhat_svi = rslds_svi.most_likely_states(xhat_svi, y)

# Uncomment this to fit with variational EM
# q_vem = SLDSTriDiagVariationalPosterior(rslds, y)
# elbos = rslds.fit(q_vem, y, method="vem", num_iters=500, initialize=False)
# xhat = q_vem.mean[0]
# zhat = rslds.most_likely_states(xhat, y)

# Plot some results
plt.figure()
plt.plot(elbos)
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("ELBO")

plt.figure()
ax1 = plt.subplot(121)
plot_trajectory(z, x, ax=ax1)
plt.title("True")
ax2 = plt.subplot(122)
plot_trajectory(zhat, xhat, ax=ax2)
plt.title("Inferred")

plt.figure(figsize=(6,6))
ax = plt.subplot(111)
lim = abs(xhat).max(axis=0) + 1
plot_most_likely_dynamics(rslds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.title("Most Likely Dynamics")

plt.show()


# test transition hessian
# M = (rslds.M,) if isinstance(rslds.M, int) else rslds.M
# inputs = [np.zeros((yt.shape[0],)+M) for yt in [y]]
# masks = [np.ones_like(yt, dtype=bool) for yt in [y]]
# tags = [None] * len([y])
# input = inputs[0]
# mask = masks[0]
# tag = tags[0]

#
# obj = lambda x, x1, E_zzp1, inputt: np.sum(E_zzp1 * np.squeeze(self.log_transition_matrices(np.vstack((x,x1)), inputt, mask, tag)))
# hess = hessian(obj)
# terms = np.array([hess(x[None,:], Ezzp1) for x, Ezzp1 in zip(data, expected_joints)])
# terms = np.array([hess(data[t], data[t+1], expected_joints[t], input[t:t+2]) for t in range(T-1)])
# obj = lambda x, x1, inputt: rslds.transitions.log_transition_matrices(np.vstack((x,x1)), inputt, mask, tag)
#
# R = rslds.transitions.Rs
# r = rslds.transitions.r
# W = rslds.transitions.Ws
# v = R*x[0] + r
# vsum = np.sum(np.exp(v))
# -( 1.0 / vsum * R.T @ np.diag(np.exp(v)) @ R - 1.0 / (vsum**2) * R.T@np.outer(np.exp(v),np.exp(v))@R)
