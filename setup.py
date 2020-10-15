#!/usr/bin/env python
import os
from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

# Only compile with OpenMP if user asks for it
USE_OPENMP = os.environ.get('USE_OPENMP', False)
print("USE_OPENMP", USE_OPENMP)

# Create the extensions. Manually enumerate the required
extensions = []

extensions.append(
    Extension('ssm.cstats',
              extra_compile_args=["-fopenmp"] if USE_OPENMP else [],
              extra_link_args=["-fopenmp"] if USE_OPENMP else [],
              language="c++",
              sources=["ssm/cstats.pyx"],
              )
)

extensions = cythonize(extensions)


setup(name='ssm',
      version='0.0.1',
      description='Bayesian learning and inference for a variety of state space models',
      author='Scott Linderman',
      author_email='scott.linderman@stanford.edu',
      url='https://github.com/slinderman/ssm',
      install_requires=['future', 'numpy', 'scipy', 'matplotlib', 'numba', 'scikit-learn', 'tqdm', 'autograd', 'seaborn'],
      packages=['ssm','ssm.extensions','ssm.extensions.mp_srslds'],
      ext_modules=extensions,
      include_dirs=[np.get_include(),],
      )
