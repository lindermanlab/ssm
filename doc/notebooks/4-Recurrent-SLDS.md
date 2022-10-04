---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Recurrent SLDS
A Recurrent Switching Linear Dynamical System (rSLDS) is a generalization of a Switching Linear Dynamical System (SLDS) in which the switches in the discrete state are allowed to depend on the value of the continuous state (hence the name recurrent). The rSLDS was developed by Linderman _et al_ in ["Bayesian Learning and Inference in Recurrent Switching Linear Dynamical Systems"](http://proceedings.mlr.press/v54/linderman17a.html).

In this notebook, we'll give an example of fitting an rSLDS from data, and show that it can learn globally non-linear dynamics. We will focus on using an rSLDS in the context of SSM, and so will skip most of the implementation-level details. For more information on implementation, see the paper above.


## 1. Generative model for rSLDS
The generative model for rSLDS is the same as the SLDS case, except that the discrete state transition probabilities are modulated by the continuous state.

1. **Discrete State Update**. At each time step, sample a new discrete state $z_t \mid z_{t-1}, x_{t-1}$ with probabilities driven by a logistic regression on the continuous state: 
$$
p(z_t = i \mid z_{t-1} = j, x_{t-1}) \propto
\exp{\left( \log (P_{j,i}) + W_i^T u_t + R_i ^T x_{t-1} \right)}
$$
where $W_i$ is a vector of weights associated with discrete state $i$, that control dependence on an external, known input $u_t$. $R_i$ is again a vector of weights assocaited with state $i$, which weights the contribution from the prior state.

2. **Continuous State Update**. Update the state using the dynamics matrix corresponding to the new discrete state:
$$
x_t = A_k x_{t-1} + V_k u_{t} + b_k + w_t
$$
$A_k$ is the dynamics matrix corresponding to discrete state $k$. $u_t$ is the input vector (specified by the user, not inferred by SSM) and $V_k$ is the corresponding control matrix. The vector $b$ is an offset vector, which can drive the dynamics in a particular direction. 
The terms $w_t$ is a noise terms, which perturbs the dynamics. 
Most commonly these are modeled as zero-mean multivariate Gaussians,
but one nice feature of SSM is that it supports many distributions for these noise terms. See the Linear Dynamical Systems notebook for a list of supported dynamics models.

3. **Emission**. We now make an observation of the state, according to the specified observation model. In the general case, the state controlls the observation via a Generalized Linear Model:
$$
y_t \sim \mathcal{P}(\eta(C_k x_t + d_k + F_k u_t + v_t))
$$
$\mathcal{P}$ is a probabibility distribution. The inner arguments form an affine measurement of the state, which is then passed through the inverse link function $\eta(\cdot)$.
In this case, $C_k$ is the measurement matrix corresponding to discrete state $k$, $d_k$ is an offset or bias term corresponding to discrete state $k$, $F_k$ is called the feedthrough matrix or passthrough matrix (it passes the input directly to the emission). In the Gaussian case, the emission can simply be written as $y_t = C_k x_t + d_k + F_k u_t + v_t$ where $v_t$ is a Gaussian r.v. See the Linear Dynamical System notebook for a list of the observation models supported by SSM.
  





```{code-cell} ipython3
import os
import pickle
import copy

import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(12345)

import matplotlib.pyplot as plt
from matplotlib import gridspec
%matplotlib inline

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import ssm
from ssm.util import random_rotation, find_permutation

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

def plot_observations(z, y, ax=None, ls="-", lw=1):

    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            ax.plot(t[start:stop + 1], y[start:stop + 1, n],
                    lw=lw, ls=ls,
                    color=colors[z[start] % len(colors)],
                    alpha=1.0)
    return ax


def plot_most_likely_dynamics(model,
    xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=20,
    alpha=0.8, ax=None, figsize=(3, 3)):
    
    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

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
```

## 2. Simulating Data from an rSLDS

Below, we create a simulated dataset from a non-linear system, which we'll call the "Nascar" dataset. The Nascar dataset is meant to emulate cars going around a track. There are 4 states total: 2 each for driving  along each straightaway, and two for semicircular turns at each end of the track.

You'll note that in creating the rSLDS we use **transitions="recurrent_only"**. This means that the transition probabilities are determined only by the previous state (and on the inputs, if present). There is no dependence on the prior $z_t$. Instead, each state simply has a constant bias $r_i$ which biases the transitions toward state $i$. This model is strictly less flexible that the full rSLDS formulation. By setting the weights on the current state to be very large, we make the discrete state transitions essentially deterministic. After creating the rSLDS and sampling a trajectory, we plot the true trajectory below.

```{code-cell} ipython3
# Global parameters
T = 10000
K = 4
D_obs = 10
D_latent = 2
```

```{code-cell} ipython3
# Simulate the "nascar" data
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
```

**Visualizing Trajectories**

We've defined some helper functions above to plot the latent state trajectories, and color code them according to the discrete state. In the left panel, we show the continuous state trajectories. In the right panel below, we show 3 of the observations over the first 1000 time steps (our observations are 10 dimensional, but we've plotted 3 observation traces to reduce clutter).

```{code-cell} ipython3
fig = plt.figure(figsize=(15, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3]) 
ax0 = plt.subplot(gs[0])
plot_trajectory(z, x, ax=ax0)
plt.title("True Trajectory")

ax1 = plt.subplot(gs[1])
plot_observations(z[:1000], y[:1000,:3], ax=ax1)
plt.title("Observations for first 1000 time steps")
plt.tight_layout()
```

## 3. Exercise

### 3.1 Linear vs. Non-Linear Systems
It's worth looking at the plot of the trajectories and considering the following: what behavior does the latent state show that could not be captured by a single linear dynamical system?

### 3.2 Understanding the discrete state transitions
Let's look again at the form of the discrete state transition probabilities:

$$
p(z_t = i \mid z_{t-1} = j, x_{t-1}) \propto
\exp{\left( \log (P_{j,i}) + w_i^T u_t + r_i ^T x_{t-1} \right)}
$$

In this case, we have used the **recurrent only** transitions class, which means we get rid of the transition matrix and replace it with a bias. Since we don't have any external inputs here, we can also leave out the input terms:


$$
p(z_t = i \mid z_{t-1} = j, x_{t-1}) \propto
\exp{\left( r_i + R_i ^T x_{t-1} \right)}
$$

What happens as the magnitude of the entries in $R_i$ become very large (compared to the entries of $R_j$ for the other states? Do the transitions become more or less random?  

## 4. Fitting an rSLDS
Below, we create a new rSLDS object and fit it to the data generated above (note that our new rSLDS will only have access to the observations $y$ and not the true states $z$ or $x$). 

### 4.1 Fitting Methods
The fitting methods available for the rSLDS are the same as those available for the SLDS. We've reproduced the section on fitting methods from the SLDS notebook below.

**Important Note:**  
 <span style="font-size:larger;">
Understanding the following section is not necessary to use SSM! _For practical purposes, it is almost always best to use the Laplace-EM method with the Structured Mean-Field Posterior, which is the default._ Running the below cells will be a bit slow on a typical laptop (around 5 minutes). We're working on speeding things up in future releases of SSM.
</span>

**Parameter Learning for rSLDS**  
Parameter learning in an rSLDS requires approximate methods. SSM provides two approximate inference algorithms: Black Box Variational Inference (`"bbvi"`) and Laplace Variational EM (`"laplace_em"`). We don't have the space to describe these methods in detail here, but Black Box Variational Inference was described in ["Variational Inference: A Review for Statisticians"](https://arxiv.org/pdf/1601.00670.pdf) by Blei et al. The Laplace Approximation is described in several sources, but a good reference for the context of state-space models is ["Estimating State and Parameters in state-space models of Spike Trains,"](https://pdfs.semanticscholar.org/a71e/bf112cabd47cc67284dc8c12ab7644195d60.pdf) a book chapter by Macke et al.  The specific method used in this notebook is described by ["Zoltowski et al (2020)"](https://arxiv.org/abs/2001.04571).



**Approximate Posterior Distributions**
When using approximate methods, we must choose the form of the distribution we use to approximate the posterior. Here, SSM provides three options:
1. `variational_posterior="meanfield"`
The mean-field approximation uses a factorized distribution as the approximating posterior. Compatible with the BBVI method.

2. `variational_posterior="tridiag"`
This approximates the posterior using a Gaussian with a block tridiagonal covariance matrix, which can be thought of as approximating the SLDS posterior with the posterior from an LDS. Compatible with the BBVI method.

3. `variational_posterior="structured_meanfield"`
This assumes a posterior where the join distribution over the continuous and discrete latent states factors as follows. If $q(z,x \mid y)$ is the joint posterior of the discrete and continuous states given the data, we use the approximation $q(z,x \mid y) \approx q(z \mid y)q(x \mid y)$, where $q(z \mid y)$ is the posterior for a Markov chain. Compatible with the Laplace-EM method.

**Calling the Fit function in SSM**  
All models in SSM share the same general syntax for fitting a model from data. Below, we call the fit function using three different methods and compare convergence. The syntax is as follows:
```python
elbos, posterior = slds.fit(data, method= "...",
                            variational_posterior="...",
                            num_iters= ...)
```
In the the call to `fit`, method should be one of {`"bbvi"`, `"laplace_em"`}.  
The `variational_posterior` argument should be one of {`"mf"`, `"structured_meanfield"`}. However, when using Laplace-EM _only_ structured mean field is supported.
Below, we fit using four methods, and compare convergence.


### 4.2 Getting the Inferred States
For every LDS, SLDS, and rSLDS model in SSM, calling `fit` returns a tuple of `(elbos, posterior)`.  `elbos` is a list containing a lower bound on the log-likelihood of the data at each iteration, used to check the convergence of the fitting algorithm. `posterior` is a posterior object (the exact type depends on which posterior is used). The posterior object is used to get an estimate of the latent variables (in this case $x$ and $z$) for each time step.

Below, we use the line:
```python
xhat_lem = q_lem.mean_continuous_states[0]
```

to get an estimate $\hat x$ of the continuous state over time. The reason for the index `[0]` is that `posterior.mean_continuous_states` will return a list, where each entry is the posterior for a given trial. In this case, we only have a single trial, so we get the first (and only) element of the list.

```{code-cell} ipython3
# Fit an rSLDS with its default initialization, using Laplace-EM with a structured variational posterior
rslds = ssm.SLDS(D_obs, K, D_latent,
             transitions="recurrent_only",
             dynamics="diagonal_gaussian",
             emissions="gaussian_orthog",
             single_subspace=True)
rslds.initialize(y)
q_elbos_lem, q_lem = rslds.fit(y, method="laplace_em",
                               variational_posterior="structured_meanfield",
                               initialize=False, num_iters=100, alpha=0.0)
xhat_lem = q_lem.mean_continuous_states[0]
rslds.permute(find_permutation(z, rslds.most_likely_states(xhat_lem, y)))
zhat_lem = rslds.most_likely_states(xhat_lem, y)

# store rslds
rslds_lem = copy.deepcopy(rslds)
```

```{code-cell} ipython3
# Fit an rSLDS with its default initialization, using BBVI with a structured variational posterior
rslds = ssm.SLDS(D_obs, K, D_latent, 
             transitions="recurrent_only",
             dynamics="diagonal_gaussian",
             emissions="gaussian_orthog",
             single_subspace=True)
rslds.initialize(y)

q_elbos_bbvi, q_bbvi = rslds.fit(y, method="bbvi",
                                 variational_posterior="meanfield",
                                 initialize=False, num_iters=1000)
```

```{code-cell} ipython3
# Get the posterior mean of the continuous states
xhat_bbvi = q_bbvi.mean[0]

# Find the permutation that matches the true and inferred states
rslds.permute(find_permutation(z, rslds.most_likely_states(xhat_bbvi, y)))
zhat_bbvi = rslds.most_likely_states(xhat_bbvi, y)
```

### 4.4 Checking Convergence
Below, we plot the ELBO obtained via both Laplace-EM and BBVI. We see that the Laplace-EM algorithm tends to converge must faster (and to a better lower bound).

```{code-cell} ipython3
# Plot some results
plt.figure()
plt.plot(q_elbos_bbvi, label="BBVI")
plt.plot(q_elbos_lem[1:], label="Laplace-EM")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("ELBO")
```

## 5. Visualizing True and Inferred States
We wrote some helper functions above that plot a state trajectory, with different colors corresponding to the discrete latent states. **Note**: we only can recover the true system up to an affine transformation. That's why, even though we have permuted the discrete states to match the true system, the colors don't always match up.

In the cell immediately below, we see that the estimated latent trajectories found using Laplace-EM match the ground-truth more closely. In the cell below that, we extract the dynamics matrices the $A_k$s and use them to plot the system dynamics in each state. Note that the Laplace-EM algorithm does a better job at finding the positions in state-space which trigger discrete state transitions. 

```{code-cell} ipython3
plt.figure(figsize=[10,4])
ax1 = plt.subplot(131)
plot_trajectory(z, x, ax=ax1)
plt.title("True")
ax2 = plt.subplot(132)
plot_trajectory(zhat_bbvi, xhat_bbvi, ax=ax2)
plt.title("Inferred, BBVI")
ax3 = plt.subplot(133)
plot_trajectory(zhat_lem, xhat_lem, ax=ax3)
plt.title("Inferred, Laplace-EM")
plt.tight_layout()
```

```{code-cell} ipython3
plt.figure(figsize=(6,4))
ax = plt.subplot(111)
lim = abs(x).max(axis=0) + 1
plot_most_likely_dynamics(true_rslds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.title("True Dynamics")

plt.figure(figsize=(6,4))
ax = plt.subplot(111)
lim = abs(xhat_lem).max(axis=0) + 1
plot_most_likely_dynamics(rslds_lem, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.title("Inferred Dynamics, Laplace-EM")

plt.figure(figsize=(6,4))
ax = plt.subplot(111)
lim = abs(xhat_bbvi).max(axis=0) + 1
plot_most_likely_dynamics(rslds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.title("Inferred Dynamics, BBVI")
```
