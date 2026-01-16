Quickstart Guide
================

This guide will walk you through the basics of the different classes of estimators and how to use them. 

Discrete estimators
-------------------

The basic form of the discrete estimators is a Python dictionary, which represents the joint probability distribution over some set of elements. 
The keys are tuples (which can be intergers, strings, tuples, anything), and the values are the probabilities.

.. code-block:: python
   
   # An example AND distribution 
   joint_distribution = {
        (0,0,0) : 1/4,
        (0,1,0) : 1/4,
        (1,0,0) : 1/4,
        (1,1,1) : 1/4,
        }

The discrete estimators generally take in indices, represented as tuples of integers that give in the indices of the subsets of elements being considered, and return two values: a dictionary of the pointwise (local) measures, and the expected values. 

.. code-block:: python

   from syntropy.discrete import (
        conditional_entropy, 
        mutual_information
        )

   lce, ce = conditional_entropy(
        idxs_x = (0,),
        idxs_y = (2,),
        joint_distribution = joint_distribution
        )
   lmi, mi = mutual_information(
        idxs_x = (0,1),
        idxs_y = (2,),
        joint_distribution = joint_distribution
        )

The pointwise dictionary keys are a tuple of tuples, indicating the states of all elements in `idxs_x` and `idxs_y` respectively.

.. code-block:: python 

   lce = {
        ((1,),(0,)) : 1.585,
        ((1,),(1,)) : 0,
        ((0,),(0,)) : 0.585
        }

   lmi = {
        ((1, 0), (0,)): 0.415,
        ((0, 0), (0,)): 0.415,
        ((1, 1), (1,)): 2.0),
        ((0, 1), (0,)): 0.415
        }

Gaussian estimators 
-------------------

The Gaussian estimators are computed from continuous, multidimensional numpy arrays. 
Unlike the discrete and KNN estimators, there are separate functions for expected and local measures - this was done because, in my cases, expected values can be computed from the covariance matrix directly, which is much more efficient than vectorized local computation

.. code-block:: python 

   import numpy as np 
   from syntropy.gaussian import (
        mutual_information, 
        local_mutual_information
        )

   num_samples = 100_000
   rand = np.random.randn(num_samples)
   data = np.zeros((3, num_samples))
   data[0,:] = rand 
   data[1,:] = 0.5 * data[0,:] + np.sqrt(1 - 0.5**2) + np.random.randn(num_samples)
   data[2,:] = -0.3 * data[0,:] + np.sqrt(1 - -0.3**2) + np.random.randn(num_samples)

   cov = np.cov(data, ddof=0.0)
   mi = mutual_information(
        idxs_x = (0,1),
        idxs_y = (2,),
        cov = cov
        )
   lmi = local_mutual_information(
        idxs_x = (0,1),
        idxs_y = (2,),
        data = data
        )

The average mutual information is a floating point value, while the local mutual information is a 1-dimensional Numpy array, with one cell for each sample

KNN estimators 
--------------

K-nearest neighbors estimators behave similarly to the local Gaussian estimators: they take in the full, multidimensional dataset, and given a `k` value, returns both the local and average values. 

.. code-block:: python

   from syntropy.knn import mutual_information 
        
   # Using the same data as above.
   num_samples = 10_000
   rand = np.random.randn(num_samples)
   data = np.zeros((3, num_samples))
   data[0,:] = rand 
   data[1,:] = 0.5 * data[0,:] + np.sqrt(1 - 0.5**2) + np.random.randn(num_samples)
   data[2,:] = -0.3 * data[0,:] + np.sqrt(1 - -0.3**2) + np.random.randn(num_samples)
   
   lmi, mi = mutual_information(
        idxs_x = (0,1),
        idxs_y = (2,),
        data = data, 
        k=k,
        algorithm = 1
   )

Once again, the `lmi` object is a Numpy array, while mi is a floating point value

Neural estimators
-----------------

The neural estimators use normalizing flows to estimate information-theoretic quantities for continuous random variables. They require PyTorch tensors as input and use the `nflows` library under the hood.

.. code-block:: python

   import torch
   from syntropy.neural import (
       differential_entropy,
       mutual_information,
       total_correlation,
       higher_order_information
   )

   # Generate sample data (samples x features format)
   num_samples = 10_000
   rand = torch.randn(num_samples)
   data = torch.zeros((num_samples, 3))
   data[:, 0] = rand
   data[:, 1] = 0.5 * data[:, 0] + torch.sqrt(torch.tensor(1 - 0.5**2)) * torch.randn(num_samples)
   data[:, 2] = -0.3 * data[:, 0] + torch.sqrt(torch.tensor(1 - 0.3**2)) * torch.randn(num_samples)

   # Compute mutual information
   ptw_mi, mi = mutual_information(
       idxs_x=(0, 1),
       idxs_y=(2,),
       data=data,
       verbose=True  # Print training progress
   )

The neural estimators return both pointwise (local) values as a tensor and the average value as a float. You can also pass separate test data using the ``data_test`` parameter for out-of-sample evaluation.

For multivariate measures like total correlation and O-information:

.. code-block:: python

   # Compute total correlation
   tc = total_correlation(
       idxs=(0, 1, 2),
       data=data,
       verbose=True
   )

   # Compute O-information, S-information, total correlation, and dual total correlation together
   results = higher_order_information(
       idxs=(0, 1, 2),
       data=data,
       verbose=True
   )
   print(results["o_information"])
   print(results["s_information"])
   print(results["total_correlation"])
   print(results["dual_total_correlation"])

Customizing the normalizing flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can customize the normalizing flow architecture and training process using ``flow_kwargs`` and ``train_kwargs``:

.. code-block:: python

   # Custom flow architecture
   flow_kwargs = {
       "num_layers": 8,         # Number of flow layers (default: 5)
       "hidden_features": 128,  # Neurons per hidden layer (default: 64)
       "dropout_probability": 0.2  # Dropout rate (default: 0.1)
   }

   # Custom training parameters
   train_kwargs = {
       "batch_size": 512,       # Batch size (default: 256)
       "lr": 1e-3,              # Learning rate (default: 1e-4)
       "num_epochs": 200,       # Training epochs (default: 100)
       "weight_decay": 1e-4,    # L2 regularization (default: 1e-5)
       "convergence_threshold": 0.01  # Early stopping threshold (default: 0.0)
   }

   ptw_mi, mi = mutual_information(
       idxs_x=(0,),
       idxs_y=(1,),
       data=data,
       flow_kwargs=flow_kwargs,
       train_kwargs=train_kwargs
   )

The default hyperparameters work reasonably well for most cases, but may need tuning for specific applications. Sample sizes of 10,000 or more typically produce reliable convergence.
