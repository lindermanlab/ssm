.. ssm documentation master file, created by
   sphinx-quickstart on Mon Oct  3 11:12:25 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ssm's documentation!
===============================

.. important::
  We're working full time on a JAX refactor of SSM that will take advantage of JIT compilation, GPU and TPU support, automatic differentation, etc. 
  Please stay tuned for updates soon!

This package has fast and flexible code for simulating, learning, and performing inference in a variety of state space models.
Currently, it supports:

#. Hidden Markov Models (HMM)
#. Auto-regressive HMMs (ARHMM)
#. Input-output HMMs (IOHMM)
#. Hidden Semi-Markov Models (HSMM)
#. Linear Dynamical Systems (LDS)
#. Switching Linear Dynamical Systems (SLDS)
#. Recurrent SLDS (rSLDS)
#. Hierarchical extensions of the above
#. Partial observations and missing data

We support the following observation models:

#. Gaussian
#. Student's t
#. Bernoulli
#. Poisson
#. Categorical
#. Von Mises

HMM inference is done with either expectation maximization (EM) or stochastic gradient descent (SGD).
For SLDS, we use stochastic variational inference (SVI).

.. toctree::

.. include::
   auto_examples/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`