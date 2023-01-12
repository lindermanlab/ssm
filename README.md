# SSM_star: Bayesian learning and inference for state space models
This is a fork of SSM.   

## Tests
The unit tests are mostly all passing.  However, 1 unit test (of 18) fails in `test_lds.py`.  In particular, `test_lds_log_probability_perf` is currently failing, with an assertion error about dimensionalities matching.  It is unclear to me why this is failing on my end, even though the build seems to be passing in the cloud.   Perhaps the build does not require unit tests to pass. 


## Installation 
Installation (to Python 3.8) required some extra steps not specified in the ssm repo.    In particular, note that for `src/`, we need numpy==1.23.5 to avoid an importing error with "SystemError: initialization of _internal failed without raising an exception."  Moreover, `tests/` require additional dependencies, such as `pyhsmm`, that also must be installed from git branches rather than from pypi. 
For more information, see `requirements.txt` and `dev-requirements.txt`.
For development, we need pylds and pyhsmm for comparison. 


**The remainder of the README is from SSM.**


**Note: We're working full time on a [JAX](https://github.com/google/jax) refactor of SSM that will take advantage of JIT compilation, GPU and TPU support, automatic differentation, etc. You can keep track of our progress [here](https://github.com/probml/ssm-jax/). We're hoping to make an official release soon!**

This package has fast and flexible code for simulating, learning, and performing inference in a variety of state space models.
Currently, it supports:

- Hidden Markov Models (HMM)
- Auto-regressive HMMs (ARHMM)
- Input-output HMMs (IOHMM)
- Hidden Semi-Markov Models (HSMM)
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
- Categorical
- Von Mises

HMM inference is done with either expectation maximization (EM) or stochastic gradient descent (SGD).  For SLDS, we use stochastic variational inference (SVI).

# Examples
Here's a snippet to illustrate how we simulate from an HMM.
```
import ssm
T = 100  # number of time bins
K = 5    # number of discrete states
D = 2    # dimension of the observations

# make an hmm and sample from it
hmm = ssm.HMM(K, D, observations="gaussian")
z, y = hmm.sample(T)
```

Fitting an HMM is simple.
```
test_hmm = ssm.HMM(K, D, observations="gaussian")
test_hmm.fit(y)
zhat = test_hmm.most_likely_states(y)
```

The notebooks folder has more thorough, complete examples of HMMs, SLDS, and recurrent SLDS.

# Installation
```
git clone https://github.com/lindermanlab/ssm
cd ssm
pip install numpy cython
pip install -e .
```
This will install "from source" and compile the Cython code for fast message passing and gradients.

To install with some parallel support via OpenMP, first make sure that your compiler supports it.  OS X's default Clang compiler does not, but you can install GNU gcc and g++ with conda.  Once you've set these as your default, you can install with OpenMP support using
```
USE_OPENMP=True pip install -e .
```
