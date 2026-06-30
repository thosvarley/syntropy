Changelog
=========

All notable changes to Syntropy are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

Added
~~~~~

* ``syntropy.mixed.mutual_information`` now accepts a ``continuous_estimator``
  argument (``"gaussian"`` or ``"knn"``, with a ``k`` parameter for the KNN
  option), matching the mixed entropy functions. The KNN estimator recovers the
  true mutual information for non-Gaussian continuous marginals, where the
  Gaussian plug-in is only an approximation.

Fixed
~~~~~

* ``syntropy.mixed`` estimators no longer raise an ``IndexError`` when given a
  single continuous variable (the covariance is now treated as a 1x1 matrix).

0.0.1
-----

Added
~~~~~

* Initial alpha release.
* Discrete, Gaussian, KNN (Kraskov), neural (normalizing-flow), and mixed estimators.
* Pointwise (local) values alongside expected values across estimators.
* Higher-order measures: total correlation, dual total correlation,
  O-information, and S-information.
* Partial information decomposition, partial entropy decomposition,
  generalized information decomposition, and integrated information decomposition.
* Information rates and Lempel-Ziv complexity measures for time series.
