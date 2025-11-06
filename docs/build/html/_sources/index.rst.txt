.. Syntropy documentation master file, created by
   sphinx-quickstart on Wed Nov  5 16:27:52 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/header.png
   :alt: Project Header
   :align: center
   :width: 100%

Syntropy documentation
======================

**Syntropy** is a Python library for multivariate information theoretic analysis of discrete and continuous data. 
It provides efficient implementations of a large variety of information measures, from basic functionns like the Shannon entropy, mutual information, and Kullback-Leibler divergence, to modern constsructs like the partial information decomposition, higher-order measures (e.g. the O-information), and information rates time series data. 

Example
-------
.. code-block:: python

   # Example with discrete distribution.
   from syntropy.discrete import mutual_information 
   from syntropy.discrete.distributions import XOR_DIST

   ptw, avg = mutual_information(
        idxs_x = (0,1), 
        idxs_y = (2,), 
        joint_distribution=XOR_DIST
        )
   print(f"I(X1,X2 ; Y) = {avg}") # Equal to 1 bit. 

   # Example with Gaussian estimator 
   import numpy as np 
   from scipy.stats import multivariate_normal
   from syntropy.gaussian import mutual_information 

   num_samples = 100_000
   cov = np.array([
        [0.99999999, 0.24404644, 0.65847509],
        [0.24404644, 0.99999985, 0.24163274],
        [0.65847509, 0.24163274, 0.99999996]
        ])
   data = multivariate_normal.rvs(
        mean = [0,0,0],
        cov = cov,
        size=num_samples
        ).T
   avg = mutual_information(
        idxs_x=(0,1),
        idxs_y=(2,), cov=np.cov(data, ddof=0.0)
        )
   print(f"I(X;Y) = {avg:.3} nat")

   # Example with Kraskov estimator 
   from syntropy.knn import mutual_information 

   num_samples = 10_000
   rand = np.random.randn(num_samples)
   data = np.vstack((rand, rand))
   data[1,:] = (data[0,:]**2) + np.random.randn(num_samples)

   ptw, avg = mutual_information(
        idxs_x=(0,),
        idxs_y=(1,),
        data=data,
        k = 5,
        algorithm = 1 # Whether to use the KSG1 or KSG2 MI estimator.
   )

   print(f"I(X;Y) = {avg:.3} nat")

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/syntropy.discrete
   api/syntropy.gaussian
   api/syntropy.knn
   api/syntropy.neural

.. toctree::
   :maxdepth: 1
   :caption: Additional Information:

   theory
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :maxdepth: 2
   :caption: Contents:

