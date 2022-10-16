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

# Multi-population recurrent switching linear dynamical systems overview


+++ {"colab_type": "text", "id": "view-in-github"}

<a href="https://colab.research.google.com/github/lindermanlab/ssm/blob/master/notebooks/Multi-Population%20rSLDS.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

+++

## If you want to quickly see how to fit your own data, jump down to the "Fit model to data" section
<br />
<br />

This notebook goes through the simulation example shown in our manuscript (Figure 2A,B).

Below, we briefly describe the model. We also recommend looking at the "Recurrent SLDS" notebook, which provides more details on the standard rSLDS.
<br />
<br />

**1. Data**.
Let $y_t^{_{(j)}}$ denote a vector of activity measurements of the $N_j$ neurons in population $j$ in time bin $t$.
<br />

**2. Emissions**. 
let $x_t^{_{(j)}}$ denote a continuous latent state of population $j$ at time $t$. The population states may differ in dimensionality~$D_j$, since populations may differ in size and complexity. The observed activity of population $j$ is modeled with a generalized linear model,
\begin{align}
    E[y_t^{(j)}] &= f(C_j x_t^{(j)} + d_j),
\end{align}
where each population has its own linear mapping parameterized by $\{C_j, d_j\}$. In this notebook, we use a Poisson GLM. Inputs can also be passed into this GLM, as described in the rSLDS notebook.

There are multi-population emissions classes that will be loaded in the example below.
<br />

**3. Continuous State Update (Dynamics)**. 
The dynamics of a switching linear dynamical system are piecewise linear, with the linear dynamics at a given time determined by a discrete state, (more on discrete states below).

\begin{align}
    x_t \sim 
    A^{(z_t)} x_{t-1} + b^{(z_t)}
\end{align} 

where $z_t$ is the discrete state, $A^{(z_t)}$ is the dynamics for that discrete state, and $x_t$ contains the latents from all populations, $[x_t^1, x_t^2, ..., x_t^J]$. We ignore the noise term here for simplicity.

Having unique continuous latents for each population allows us to decompose the dynamics in an interpretable manner. We model the temporal dynamics of the continuous states as
\begin{align}
    x_t^{(j)} \sim
    A_{(j \: to \: j)}^{(z_t)} x_{t-1}^{(j)} 
    + \sum_{i \neq j} A_{(i \: to \: j)}^{(z_t)} x_{t-1}^{(i)} 
    + b_j^{(z_t)}.
\end{align} 

In the full dynamics matrices, $A^{(z_t)}$ we will show in the example below, the on-diagonal blocks represent the internal dynamics, $A_{(j \: to \: j)}^{(z_t)}$ and the off-diagonal blocks represent the external dynamics, $A_{(i \: to \: j)}^{(z_t)}$.


**4. Discrete State Update (Transitions)**. 
Recurrent transitions are based on the continuous latent state. Our recurrent transitions have a sticky component, $S$ that determines the probabilities of staying in a state, and a switching component, $R$, that determines the probabilities of switching to states. In the model we use in this notebook:

\begin{align}
    p(z_t = i \mid z_{t-1} = j, x_{t-1}) &= \mathrm{softmax}\bigg\{ \Big( \big(R x_{t-1}\big) + r\Big) \odot (1 - e_{z_{t-1}})  + \Big( \big(S x_{t-1} \big) + s \Big) \odot e_{z_{t-1}}  \bigg\},
\end{align}

where $e_{z_{t-1}} \in \{0,1\}^K$ is a one-hot encoding of $z_{t-1}$.

To understand which populations are contributing to the transitions, we can decompose this equation:


\begin{align}
    p(z_t = i \mid z_{t-1} = j, x_{t-1}) &= \mathrm{softmax}\bigg\{ \Big( \sum_{j=1}^J \big(R_j x_{t-1}^{(j)}\big) + r\Big) \odot (1 - e_{z_{t-1}})  + \Big( \sum_{j=1}^J \big(S_j x_{t-1}^{(j)} \big) + s \Big) \odot e_{z_{t-1}}  \bigg\},
\end{align}
where, for example, $R_j x_{t-1}^{(j)}$ contains the contribution of population $j$ towards switching to each state.


Additionally, we can include a dependency on the previous discrete state. This is included in the code package, but is not used in the example below.

\begin{align}
    p(z_t = i \mid z_{t-1} = j, x_{t-1}) &= \mathrm{softmax}\bigg\{ \log(P_{j,i}) +  \big(R x_{t-1}\big) \odot (1 - e_{z_{t-1}})  + \big(S x_{t-1} \big) \odot e_{z_{t-1}}  \bigg\},
\end{align}

There are sticky multi-population emissions classes that will be loaded in the example below.
<br />

**5. Model fitting**. 
We fit the model with variational laplace EM - see the "Variational Laplace EM for SLDS Tutorial" for more information.

+++ {"colab_type": "text", "id": "8OzC8q4bRFQv"}

## Import packages, including multipopulation extensions

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 581
colab_type: code
id: ruUnNqi5RZqT
outputId: 228b6c8e-c064-46c2-ce57-9ad88daca5c8
---
try:
    import ssm
except:
    !pip install git+https://github.com/lindermanlab/ssm.git#egg=ssm
    import ssm
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 71
colab_type: code
id: zDn3tEJhRFQv
outputId: 2f1ca1d0-8f17-404a-897f-57b8c5d353cb
---
#### General packages

from matplotlib import pyplot as plt
%matplotlib inline
import autograd.numpy as np
import autograd.numpy.random as npr

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
sns.set_style('ticks',{"xtick.major.size":8,
"ytick.major.size":8})
from ssm.plots import gradient_cmap, white_to_color_cmap

color_names = [
    "purple",
    "red",
    "amber",
    "faded green",
    "windows blue",
    "orange"
    ]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: 0rq19iIQRFQy

#### SSM PACKAGES ###

import ssm
from ssm.variational import SLDSMeanFieldVariationalPosterior, SLDSTriDiagVariationalPosterior, \
    SLDSStructuredMeanFieldVariationalPosterior
from ssm.util import random_rotation, find_permutation, relu

#Load from extensions
from ssm.extensions.mp_srslds.emissions_ext import GaussianOrthogonalCompoundEmissions, PoissonOrthogonalCompoundEmissions
from ssm.extensions.mp_srslds.transitions_ext import StickyRecurrentOnlyTransitions, StickyRecurrentTransitions
```

+++ {"colab_type": "text", "id": "Ty3EOi8bRFQ1"}

## Simulate (somewhat realistic) data

+++ {"colab_type": "text", "id": "QxDoYCRDRFQ2"}

### Set parameters of simulation

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: dalqY6zvRFQ2

K=3 #Number of discrete states

num_gr=3 #Number of populations
num_per_gr=5 #Number of latents per population
neur_per_gr=75 #Number of neurons per population

t_end=3000 #number of time bins
num_trials=1 #number of trials
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
colab_type: code
id: OHTSTNbTRFQ4
outputId: fd3b833b-8df2-425b-d3ce-103cf42d5153
---
np.random.seed(108) #To create replicable dynamics

alphas=.03+.1*np.random.rand(K) #Determines the distribution of values in the dynamics matrix, for each discrete state
print('alphas:', alphas)

sparsity=.33 #Proportion of non-diagonal blocks in the dynamics matrix that are 0

e1=.1 #Amount of noise in the dynamics
```

+++ {"colab_type": "text", "id": "t91lYSkPRFQ7"}

### Get new emissions and transitions classes for the simulated data

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: tQx540b6RFQ8

#Vector containing number of latents per population
D_vec=[]
for i in range(num_gr):
    D_vec.append(num_per_gr) 

#Vector containing number of neurons per population
N_vec=[]
for i in range(num_gr):
    N_vec.append(neur_per_gr)

D=np.sum(D_vec)
num_gr=len(D_vec)
D_vec_cumsum = np.concatenate(([0], np.cumsum(D_vec)))

#Get new multipopulation emissions class for the simulation

# gauss_comp_emissions=GaussianOrthogonalCompoundEmissions(N=np.sum(N_vec),K=1,D=np.sum(D_vec),D_vec=D_vec,N_vec=N_vec)
poiss_comp_emissions=PoissonOrthogonalCompoundEmissions(N=np.sum(N_vec),K=1,D=np.sum(D_vec),D_vec=D_vec,N_vec=N_vec,link='softplus')

#Get transitions class
true_sro_trans=StickyRecurrentOnlyTransitions(K=K,D=np.sum(D_vec)) 
```

+++ {"colab_type": "text", "id": "DDu2GnRGRFQ-"}

### Create simulated data

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: VLN8FWLLRFQ_

np.random.seed(10) #To create replicable simulations

A_masks=[]

A_all=np.zeros([K,D,D]) #Initialize dynamics matrix
b_all=np.zeros([K,D]) #Initialize dynamics offset


#Create initial ground truth model, that we will modify
true_slds = ssm.SLDS(N=np.sum(N_vec),K=K,D=int(np.sum(D_vec)),
             dynamics="gaussian",
             emissions=poiss_comp_emissions,
             transitions=true_sro_trans)

#Create ground truth transitions
v=.2+.2*np.random.rand(1)
for k in range(K):
    inc=np.copy(k)
    true_slds.transitions.Rs[k,D_vec_cumsum[inc]:D_vec_cumsum[inc]+1]=v
    true_slds.transitions.Ss[k,D_vec_cumsum[inc]:D_vec_cumsum[inc]+1]=v-.1

true_slds.transitions.r=0*np.ones([K,1])
true_slds.transitions.s=5*np.ones([K,1])

#Create ground truth dynamics for each state
for k in range(K):

    ##Create dynamics##
    alpha=alphas[k]

    A_mask=np.random.rand(num_gr,num_gr)>sparsity #Make some blocks of the dynamics matrix 0

    A_masks.append(A_mask)

    for i in range(num_gr): 
        A_mask[i,i]=1

    A0=np.zeros([D,D])
    for i in range(D-1):
        A0[i,i+1:]=alpha*np.random.randn(D-1-i)
    A0=(A0-A0.T)

    for i in range(num_gr):
        A0[D_vec_cumsum[i]:D_vec_cumsum[i+1],D_vec_cumsum[i]:D_vec_cumsum[i+1]]=2*A0[D_vec_cumsum[i]:D_vec_cumsum[i+1],D_vec_cumsum[i]:D_vec_cumsum[i+1]]


    A0=A0+np.identity(D)
    A=A0*np.kron(A_mask, np.ones((num_per_gr, num_per_gr)))

    A=A/(np.max(np.abs(np.linalg.eigvals(A)))+.01) #.97

    b=1*np.random.rand(D)

    A_all[k]=A
    b_all[k]=b

true_slds.dynamics.As=A_all
true_slds.dynamics.bs=b_all


zs, xs, _ = true_slds.sample(t_end) #Sample discrete and continuous latents from model for simulation

#Get spike trains that have an average firing rate of 0.25 per bin
tmp=np.mean(relu(np.dot(true_slds.emissions.Cs[0],xs.T)+.1*true_slds.emissions.ds[0][:,None]).T)
mult=.25/tmp
lams=relu(mult*np.dot(true_slds.emissions.Cs[0],xs.T)+.1*true_slds.emissions.ds[0][:,None]).T
ys=np.random.poisson(lams) #Get spiking activity based on poisson statistics
```

+++ {"colab_type": "text", "id": "twuPg8wRRFRC"}

## Plot simulated data

+++ {"colab_type": "text", "id": "1-VkH7xSRFRC"}

### Dynamics matrices ($A^z$)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 797
colab_type: code
id: qv7hO5fgRFRD
outputId: 7b2a3151-a5dc-4ac0-c446-ecbb212ffb61
---
# vmin,vmax=[-1,1]
vmin,vmax=[-.5,.5] #zoom in to see colors more clearly


for k in range(K):
    
    plt.figure(figsize=(4,4))
    plt.imshow(true_slds.dynamics.As[k], aspect='auto', interpolation="none", vmin=vmin, vmax=vmax, cmap='RdBu')
    offset=-.5
    for nf in D_vec:        
        plt.plot([-0.5, D-0.5], [offset, offset], '-k')
        plt.plot([offset, offset], [-0.5, D-0.5], '-k')
        offset += nf
    plt.xticks([])
    plt.yticks([])
    plt.title('Actual State '+str(k))
```

+++ {"colab_type": "text", "id": "-Y7R3y6TRFRF"}

### Discrete states ($z$)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 177
colab_type: code
id: Wmk2_uI-RFRG
outputId: 2f434446-fbaf-4a00-d186-81a557b35011
---
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(zs[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.xlim(0, t_end)
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])
```

+++ {"colab_type": "text", "id": "ufQwDDKTRFRI"}

### Transitions (in a shorter time window)
The contribution of population $j$ to staying in a state is $S_j x^{j}$ and the contribution to switching to a state is $R_j x^{j}$

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: fp1QQ_7dRFRI

dur=200
st_t=650
end_t=st_t+dur
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 310
colab_type: code
id: -FWpoP7-RFRK
outputId: bcc60411-fb9e-44d0-bc16-e382fe7919aa
---
plt.figure(figsize=(8, 4))

j=0

plt.subplot(211)
for g in range(K):
    plt.plot(np.dot(xs[st_t:end_t,D_vec_cumsum[g]:D_vec_cumsum[g+1]],true_slds.transitions.Rs[j,D_vec_cumsum[g]:D_vec_cumsum[g+1]].T))
plt.xlim(0, dur)
plt.ylabel('Contribution towards \n switching to \n purple state',rotation=60)
plt.xticks([])
plt.yticks([])

j=1
plt.subplot(212)
for g in range(K):
    plt.plot(np.dot(xs[st_t:end_t,D_vec_cumsum[g]:D_vec_cumsum[g+1]],true_slds.transitions.Ss[j,D_vec_cumsum[g]:D_vec_cumsum[g+1]].T))
plt.xlim(0, dur)
plt.ylabel('Contribution towards \n staying in red state',rotation=60)
plt.legend(['Pop 1','Pop 2','Pop 3'])
plt.yticks([])
```

+++ {"colab_type": "text", "id": "wOO48DPARFRN"}

### Continuous latents ($x$) and spikes ($y$) for an example population

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 449
colab_type: code
id: YllKBbdhRFRN
outputId: dc7f9218-0100-4263-8501-9d2fb1696a9e
---
plt.figure(figsize=(8, 4))

plt.subplot(211)
plt.plot(xs[st_t:end_t,:num_per_gr]) #Show latents of first group
plt.xticks([])

plt.subplot(212)
plt.plot(ys[st_t:end_t,:10]) #Show first 10 neurons
```

+++ {"colab_type": "text", "id": "OZ_iLjiyRFRV"}

## Fit model to data

+++ {"colab_type": "text", "id": "g6QfY0j-RFRV"}

### To create the emissions classes for the multipopulation models, we need vectors containing the number of continuous latents per population ("D_vec") and neurons per population ("N_vec")

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: kVZDXyIiRFRW

num_gr=3 #Number of populations
num_per_gr=5 #Number of latents per population
neur_per_gr=75 #Number of neurons per population

#Vector containing number of latents per population
D_vec=[]
for i in range(num_gr):
    D_vec.append(num_per_gr) 

#Vector containing number of neurons per population
N_vec=[]
for i in range(num_gr):
    N_vec.append(neur_per_gr)
```

+++ {"colab_type": "text", "id": "_I-JFqbHRFRZ"}

#### Now create the multipopulation emissions and transitions classes for our model

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: dhnavgIgRFRa

#Get new multipopulation emissions class
poiss_comp_emissions=PoissonOrthogonalCompoundEmissions(N=np.sum(N_vec),K=1,D=np.sum(D_vec),D_vec=D_vec,N_vec=N_vec,link='softplus')

#Get new transitions class
sro_trans=StickyRecurrentOnlyTransitions(K=K,D=np.sum(D_vec), l2_penalty_similarity=10, l1_penalty=10) 
#The above l2 penalty is on the similarity between R and S (its assuming the activity to switch into a state is similar to activity to stay in a state)
#The L1 penalty is on the entries of R and S
```

Note that another new emissions class is "GaussianOrthogonalCompoundEmissions" <br />

Note that another new transitions class is "StickyRecurrentTransitions"

+++ {"colab_type": "text", "id": "syHDAea_RFRc"}

#### Now declare and fit the model

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 166
  referenced_widgets: [36b4e6c4b2064572a4412f8bc298caeb, 97247a689ff744689528aa9555c13c62,
    581b2c1a39eb439581308a5cdea07389, e8e6a6b80ed14b77900e87e1c779f459, 2e200ca5be3141f2b2ee041ec8b08e49,
    f4e534f4d6ac45c6a7da04be7d341968, ced98407e9ee4ee4bd0baaa24d4765b9, 14de5bf9b23f47d4867e135cf156ac97,
    6bd4692a8c95492d84f6353069342d4f, 62f38f8736314a098f91d7486b5b84f6, d7ffd5a5a8874c2d8e593cb29c8f14b2,
    2ddbbe21934547db9e365d905ea1f024, 19ad77a391f7459f811262d5bdbeb783, 5da21c8eac7041898c7a468f0388e632,
    eb1c51978e0f4b79968cf6618a959007, c0204fcad9f64dd48580f7d17767abc6]
colab_type: code
id: n7WRqQufRFRd
outputId: 28ddbaaa-221c-4ca8-db59-b76bbeff1dd0
---
K=3 #Number of discrete states

rslds = ssm.SLDS(N=np.sum(N_vec),K=K,D=np.sum(D_vec),
             dynamics="gaussian",
             emissions=poiss_comp_emissions,
             transitions=sro_trans,
             dynamics_kwargs=dict(l2_penalty_A=100)) #Regularization on the dynamics matrix

q_elbos_ar, q_ar = rslds.fit(ys, method="laplace_em",
                             variational_posterior="structured_meanfield", 
                             continuous_optimizer='newton',
                             initialize=True, 
                             num_init_restarts=10,
                             num_iters=30, 
                             alpha=0.25)
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: Oi8-3NcxRFRf
:outputId: 98075512-a594-4bae-fcf3-550affe796a6

plt.plot(q_elbos_ar[1:])
plt.xlabel("Iteration")
plt.ylabel("ELBO")
```

## Align solution with simulation for plotting

```{code-cell} ipython3
#The recovered discrete states can be permuted in any way. 
#Find permutation to match the discrete states in the model and the ground truth
z_inferred=rslds.most_likely_states(q_ar.mean_continuous_states[0],ys)
rslds.permute(find_permutation(zs, z_inferred))
z_inferred2=rslds.most_likely_states(q_ar.mean_continuous_states[0],ys)
```

```{code-cell} ipython3
#Each population's latents can be multiplied by an arbitrary rotation matrix
#Additionally, there may be a change in scaling between the simulation ground truth and recovered latents,
#because the simulation didn't constrain the effective emissions (C) matrix to be orthonormal like in the model

from sklearn.linear_model import LinearRegression

R=np.zeros([D,D])
for g in range(num_gr):
    lr=LinearRegression(fit_intercept=False)
    lr.fit(q_ar.mean_continuous_states[0][:,D_vec_cumsum[g]:D_vec_cumsum[g+1]],xs[:,D_vec_cumsum[g]:D_vec_cumsum[g+1]])
    R[D_vec_cumsum[g]:D_vec_cumsum[g+1],D_vec_cumsum[g]:D_vec_cumsum[g+1]]=lr.coef_
```

+++ {"colab_type": "text", "id": "0CjkFH63RFRh"}

## Plot results

+++ {"colab_type": "text", "id": "HDlMQx4ZRFRh"}

### Discrete states ($z$)

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: TGk3zoBVRFRi
:outputId: d1cebd3c-17f7-4e48-a6d3-0b9f34fbd3af

plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(zs[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.xlim(0, t_end)
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])

plt.subplot(212)
plt.imshow(z_inferred2[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.xlim(0, t_end)
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.tight_layout()
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: HRPr3DJlRFRj
:outputId: 1d6b402b-fa6d-4040-f4bf-4e4c2188c8b8

print('Discrete state accuracy: ', np.mean(zs==z_inferred2))
```

+++ {"colab_type": "text", "id": "SX1go1AyRFRl"}

#### Shorter time window

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: UhjVHiYzRFRm
:outputId: 5d4cf636-9055-46fe-af2b-d51531c97317

plt.figure(figsize=(4, 2))
plt.subplot(211)
plt.imshow(zs[None,st_t:end_t], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.xlim(0, dur)
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])
plt.xticks([])

plt.subplot(212)
plt.imshow(z_inferred2[None,st_t:end_t], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
plt.xlim(0, dur)
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
```

+++ {"colab_type": "text", "id": "e4Cz1o8GRFRo"}

### Dynamics matrices ($A^z$)

+++

We show the A matrix from when the continuous latents are aligned to ground truth, demonstrating the ability to recover the ground truth dynamics.

We also show the original recovered A matrix, which demonstrates that we can learn about the block structure, regardless of scaling/rotations.

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: 0D2QHaLRRFRo
:outputId: 891cf1a1-4aea-4c5f-f2d5-9895478b5361

plt.figure(figsize=(12, 12))

q=1

for k in range(K):
    
    plt.subplot(3,3,q)
    plt.imshow(true_slds.dynamics.As[k], aspect='auto', interpolation="none", vmin=-.5, vmax=.5, cmap='RdBu')
    offset=-.5
    for nf in D_vec:        
        plt.plot([-0.5, D-0.5], [offset, offset], '-k')
        plt.plot([offset, offset], [-0.5, D-0.5], '-k')
        offset += nf
    plt.xticks([])
    plt.yticks([])
    plt.title('Actual State '+str(k))
    
    q=q+1

    plt.subplot(3,3,q)
#     plt.imshow(rslds.dynamics.As[k], aspect='auto', interpolation="none", vmin=-.5, vmax=.5, cmap='RdBu')
    plt.imshow(R@rslds.dynamics.As[k]@np.linalg.inv(R), aspect='auto', interpolation="none", vmin=-.5, vmax=.5, cmap='RdBu')

    offset=-.5
    for nf in D_vec:        
        plt.plot([-0.5, D-0.5], [offset, offset], '-k')
        plt.plot([offset, offset], [-0.5, D-0.5], '-k')
        offset += nf
    plt.xticks([])
    plt.yticks([])
    plt.title('Aligned Predicted State '+str(k))
#     plt.savefig(folder+'dyn_est'+str(k)+'.pdf')    

    q=q+1


    plt.subplot(3,3,q)
#     plt.imshow(rslds.dynamics.As[k], aspect='auto', interpolation="none", vmin=-.5, vmax=.5, cmap='RdBu')
    plt.imshow(rslds.dynamics.As[k], aspect='auto', interpolation="none", vmin=-.5, vmax=.5, cmap='RdBu')

    offset=-.5
    for nf in D_vec:        
        plt.plot([-0.5, D-0.5], [offset, offset], '-k')
        plt.plot([offset, offset], [-0.5, D-0.5], '-k')
        offset += nf
    plt.xticks([])
    plt.yticks([])
    plt.title('Raw Predicted State '+str(k))
#     plt.savefig(folder+'dyn_est'+str(k)+'.pdf')
    
    q=q+1
    
    

        
```

+++ {"colab_type": "text", "id": "b14UYWuMRFRr"}

### Transitions
The contribution of population $j$ to staying in a state is $S_j x^{j}$ and the contribution to switching to a state is $R_j x^{j}$

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: hja_9YLPRFRr
:outputId: 6eb42921-5277-43ea-b14b-2366d2128509

plt.figure(figsize=(15, 4))


### Actual

j=0

plt.subplot(221)
for g in range(K):
    plt.plot(np.dot(xs[st_t:end_t,D_vec_cumsum[g]:D_vec_cumsum[g+1]],true_slds.transitions.Rs[j,D_vec_cumsum[g]:D_vec_cumsum[g+1]].T))
plt.xlim(0, dur)
plt.ylabel('Contribution towards \n switching to \n purple state',rotation=60)
plt.xticks([])
plt.yticks([])
plt.title('Actual')

j=1
plt.subplot(223)
for g in range(K):
    plt.plot(np.dot(xs[st_t:end_t,D_vec_cumsum[g]:D_vec_cumsum[g+1]],true_slds.transitions.Ss[j,D_vec_cumsum[g]:D_vec_cumsum[g+1]].T))
plt.xlim(0, dur)
plt.ylabel('Contribution towards \n staying in red state',rotation=60)
plt.legend(['Pop 1','Pop 2','Pop 3'])
plt.yticks([])




### Predicted

j=0

plt.subplot(222)
for g in range(K):
    plt.plot(np.dot(q_ar.mean_continuous_states[0][st_t:end_t,D_vec_cumsum[g]:D_vec_cumsum[g+1]],rslds.transitions.Rs[j,D_vec_cumsum[g]:D_vec_cumsum[g+1]].T))
plt.xlim(0, dur)
# plt.ylabel('Contribution towards \n switching to \n purple state',rotation=60)
plt.xticks([])
plt.yticks([])
plt.title('Predicted')

j=1
plt.subplot(224)
for g in range(K):
    plt.plot(np.dot(q_ar.mean_continuous_states[0][st_t:end_t,D_vec_cumsum[g]:D_vec_cumsum[g+1]],rslds.transitions.Ss[j,D_vec_cumsum[g]:D_vec_cumsum[g+1]].T))
plt.xlim(0, dur)
# plt.ylabel('Stay in Red')
# plt.xticks([])
plt.yticks([])
```

+++ {"colab_type": "text", "id": "yz4GSxjrRFRt"}

### Example fit of neural activity ($y$)

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: cgEffzVKRFRt
:outputId: 10d64661-8fbd-4259-96fc-134c35980b88

preds=rslds.smooth(q_ar.mean_continuous_states[0],ys) #get predictions

nrn=0 #Example neuron
plt.plot(ys[st_t:end_t,nrn],alpha=.5) #true spiking activity
plt.plot(lams[st_t:end_t,nrn]) #true firing rate
plt.plot(preds[st_t:end_t,nrn]) #predicted firing

plt.legend(['True Spikes','True FR','Predicted FR'])
```
