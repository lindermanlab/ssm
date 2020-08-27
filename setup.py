#!/usr/bin/env python
import os
from distutils.core import setup

setup(name='ssm',
      version='0.9',
      description='Bayesian learning and inference for a variety of state space models',
      author='Scott Linderman',
      author_email='scott.linderman@stanford.edu',
      url='https://github.com/lindermanlab/ssm',
      install_requires=['numpy', 'scipy', 'matplotlib', 'numba', 'scikit-learn', 'tqdm', 'autograd', 'seaborn', 'jax'],
      packages=['ssm', 'ssm.distributions', 'ssm.hmm'],
    )
