Quick start guide
=================

This guide will help you get started with the oars project.

Installation

To install oars with the packages required to run all of the examples, run the following command:

.. code-block:: bash

    git clone https://github.com/peterbarkley/oars.git
    pip install .[all]

Demo

View the demo notebook output `here <_static/paper.html>`_ or run the following command to generate the demo notebook:

.. code-block:: bash

    jupyter nbconvert --to html oars/examples/paper.ipynb 

Usage

The package is organized into a few main modules: matrices, algorithms, PEP, and utils. The matrices module contains functions for generating frugal resolvent splitting design matrices with specific properties. The algorithms module contains functions for solving optimization problems using the design matrices. The PEP module contains functions for solving the PEP problem to find the contraction factor for each algorithm design and to find step sizes and design matrices which optimize that contraction factor. The utils module contains functions for generating plots and metrics, as well as a repository of resolvent functions for testing. 

We use the MOSEK and GUROBI solvers by default for the optimization problems. To use the MOSEK solver, you must have a valid license and the MOSEK Python API installed. To use the GUROBI solver, you must have a valid license and the GUROBI Python API installed. With different solvers installed, you may need to change the solver parameter in the optimization functions.

The following examples demonstrate how to use the package:

.. code-block:: python

    import oars
    import numpy as np

    # Matrices for Algorithm Design 
    # Get the minimum spectral difference design matrices
    n = 6
    Z, W = oars.matrices.getMinSpectralDifference(n)

    # Get the minimum second-largest eigenvalue design matrices
    Z, W = oars.matrices.getMinSLEM(n)

    # Get the minimum total effective resistance design matrices
    Z, W = oars.matrices.getMinResist(n)

    # Get the maximum algebraic connectivity design matrices
    Z, W = oars.matrices.getMaxConnectivity(n)

    # Get d-block design matrices
    d = 2
    Z, W = oars.matrices.getBlockMin(n, n//d, builder=oars.matrices.getMinSLEM)

    # Get minimum iteration time design matrices
    t = np.random.rand(n)
    l = np.random.rand((n, n))/10
    Z, W = oars.matrices.getMinIteration(n, t=t, l=l)

    # Solve Optimization Problems
    # This minimizes the squared error between the data and the decision value x 
    # (which should return the mean)
    from oars.utils import proxs
    np.random.seed(0)
    data = np.random.rand(n)
    resolvents = [proxs.quadprox] * n
    x, results = oars.solve(n, data, resolvents, W=W, Z=Z)
    assert(np.isclose(x, np.mean(data)))

    # Get the contraction factor for a given design matrix and step size
    # for the iterates in algorithm formulation (11) in the paper
    ls = np.ones(n)*1.1
    mus = np.ones(n)*0.9
    M = oars.matrices.getMfromWCholesky(W)
    tau = oars.pep.getReducedContractionFactor(Z, M, ls=ls, mus=mus, gamma=0.5)

    # Get the optimal step size for the iterates in algorithm formulation (11)
    tau, gamma = oars.pep.getReducedContractionOptGamma(Z, M, ls=ls, mus=mus)

