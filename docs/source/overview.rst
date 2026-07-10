Overview
========

This page gives a compact, high-level map of what Syntropy can do and which
estimator to reach for, before diving into the :doc:`quickstart`.

How the package is organized
----------------------------

Syntropy is broken into two main arms — **discrete** estimators (which operate on
multivariate probability distributions represented as Python dictionaries) and
**continuous** estimators (covariance- and sample-based) — together with a small
number of additional estimator families. Within each arm, functionality is
grouped into sub-modules:

* ``shannon`` — basic Shannon quantities (entropy, conditional entropy, mutual
  information, Kullback-Leibler divergence, etc.).
* ``multivariate_mi`` — higher-order generalizations of mutual information
  (total correlation, dual total correlation, O-information, S-information).
* ``decompositions`` — the partial information decomposition (PID), partial
  entropy decomposition (PED), generalized information decomposition (GID), and
  integrated information decomposition.
* ``temporal`` — functions for time-series analysis (information rates,
  Lempel-Ziv complexity).
* ``utils`` — basic operations on discrete and Gaussian probability
  distributions, plus example distributions of theoretical note.

Estimator families
------------------

Syntropy provides several estimators so you can choose the most appropriate tool
for your data, rather than transforming the data to fit the estimator:

* **Discrete** (:mod:`syntropy.discrete`) — exact computation on a joint
  probability distribution supplied as a dictionary.
* **Gaussian** (:mod:`syntropy.gaussian`) — parametric estimation from a
  covariance matrix or continuous samples.
* **KNN / Kraskov** (:mod:`syntropy.knn`) — non-parametric estimation for
  continuous data via k-nearest-neighbors (KSG).
* **Neural** (:mod:`syntropy.neural`) — normalizing-flow estimators for complex,
  high-dimensional continuous distributions (optional ``neural`` extra).
* **Mixed** (:mod:`syntropy.mixed`) — mutual information between discrete and
  continuous variables.

Available measures
------------------

The table below summarizes which measures are currently implemented for each
estimator family. A checkmark (✓) indicates the measure is available.

.. csv-table:: Available measures by estimator
   :header: "Measure", "Discrete", "Gaussian", "KNN", "Neural", "Mixed"
   :widths: 34, 12, 12, 10, 10, 10

   "Entropy", "✓", "✓", "✓", "✓", "✓"
   "Conditional entropy", "✓", "✓", "✓", "✓", "✓"
   "Mutual information", "✓", "✓", "✓", "✓", "✓"
   "Conditional mutual information", "✓", "✓", "✓", "✓", ""
   "Kullback-Leibler divergence", "✓", "✓", "✓", "", ""
   "Total correlation", "✓", "✓", "✓", "✓", ""
   "Dual total correlation", "✓", "✓", "✓", "✓", ""
   "O-information", "✓", "✓", "✓", "✓", ""
   "S-information", "✓", "✓", "✓", "✓", ""
   "Co-information", "✓", "✓", "", "", ""
   "TSE complexity", "✓", "✓", "", "", ""
   "Partial information decomposition", "✓", "✓", "", "", ""
   "Partial entropy decomposition", "✓", "✓", "", "", ""
   "Generalized information decomposition", "✓", "✓", "", "", ""
   "Integrated (Φ) information decomposition", "✓", "✓", "", "", ""
   "Information rates", "✓", "✓", "", "", ""
   "Connected information", "✓", "", "", "", ""
   "α-synergy decomposition", "✓", "", "", "", ""
   "I_dep decomposition", "", "✓", "", "", ""

Optimizations and utilities
---------------------------

Beyond the estimators above, Syntropy includes a number of optimization
algorithms and helpers, for example:

* finding optimally-synergistic submatrices of a covariance matrix,
* finding the maximum-entropy discrete distribution consistent with given
  k-order marginals.

The ``utils`` modules also provide a range of functions for constructing and
manipulating discrete and continuous probability distributions.

For worked examples of these measures in action, see the :doc:`quickstart`.
