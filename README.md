# SSM: Bayesian learning and inference for state space models 

This package has fast and flexible code for simulating, learning, and performing inference in a variety of state space models. 
Currently, it supports:

- Hidden Markov Models (HMM)
- Auto-regressive HMMs (ARHMM)
- Input-output HMMs (IOHMM)
- Linear Dynamical Systems (LDS)
- Switching Linear Dynamical Systems (SLDS)
- Recurrent SLDS (rSLDS)
- Hierarchical extensions of the above
- Partial observations and missing data

We support the following observation models:

- Gaussian
- Student's t
- Bernoulli
- Poisson

HMM inference is done with either expectation maximization (EM) or stochastic gradient descent (SGD).  For SLDS, we use stochastic variational inference (SVI). 

# Examples
Here's a snippet to illustrate how we simulate from an HMM.
```
from ssm.models import HMM
T = 100  # number of time bins
K = 5    # number of discrete states
D = 2    # dimension of the observations

# make an hmm and sample from it
hmm = HMM(K, D, observations="gaussian")
z, y = hmm.sample(T)
```

Fitting an HMM is simple. 
```
test_hmm = HMM(K, D, observations="gaussian")
test_hmm.fit(y)
zhat = test_hmm.most_likely_states(y)
```

The notebooks folder has more thorough, complete examples of HMMs, SLDS, and recurrent SLDS.  

# Installation
```
git clone git@github.com:slinderman/ssm.git
cd ssm
pip install -e .
```
This will install "from source" and compile the Cython code for fast message passing and gradients.
