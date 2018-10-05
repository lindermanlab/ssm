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
    Extension('ssm.messages',
              extra_compile_args=[],
              extra_link_args=[],
              language="c++",
              sources=["ssm/messages.pyx"],
              )
)

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
      description='State space models in python',
      author='Scott Linderman',
      install_requires=['numpy', 'scipy', 'matplotlib'],
      packages=['ssm'],
      ext_modules=extensions,
      include_dirs=[np.get_include(),],
      )
