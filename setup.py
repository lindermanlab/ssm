#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='ssm',
      version='0.0.1',
      description='State space models in python',
      author='Scott Linderman',
      install_requires=['numpy', 'scipy', 'matplotlib'],
      packages=['ssm'],
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
      )
