#! /usr/bin/env python
import os
import numpy as np
from Cython.Build import cythonize
from setuptools.extension import Extension
from setuptools import setup, find_packages

# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create the extensions. Manually enumerate the required
# Only compile with OpenMP if user asks for it
USE_OPENMP = os.environ.get('USE_OPENMP', False)
print("USE_OPENMP", USE_OPENMP)
extensions = []
extensions.append(
    Extension('ssm.cstats',
              extra_compile_args=["-fopenmp"] if USE_OPENMP else [],
              extra_link_args=["-fopenmp"] if USE_OPENMP else [],
              language="c++",
              sources=["ssm/cstats.pyx"],
              include_dirs=[np.get_include()]
              )
)

DISTNAME = 'ssm'
DESCRIPTION = 'Bayesian learning and inference for a variety of state space models'
with open('README.md') as fp:
    LONG_DESCRIPTION = fp.read()
MAINTAINER = 'Scott Linderman'
MAINTAINER_EMAIL = 'scott.linderman@stanford.edu'
URL = ''
LICENSE = 'MIT'
DOWNLOAD_URL = ''
VERSION = '0.0.1'


if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 3.7',
          ],
          packages=find_packages(),
          ext_modules = cythonize(extensions)
          )
