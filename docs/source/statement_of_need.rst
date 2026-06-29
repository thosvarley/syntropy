Statement of Need
==================

Information theory has emerged as a kind of *lingua franca* for the study of
complex systems, finding applications in neuroscience, climatology,
developmental biology, economics, sociology, and beyond. Despite this broad
reach, the software landscape remains fragmented: most existing packages are
restricted to a single subset of analyses or a single type of data.

For example, the `DIT <https://github.com/dit/dit>`_ package is specific to
discrete information theory and uses a customized distribution object. The
`IDTxl <https://github.com/pwollstadt/IDTxl>`_ and
`JIDT <https://github.com/jlizier/jidt>`_ packages implement several classes of
estimators, but are focused specifically on information dynamics and time-series
analysis.

Multivariate information decomposition in particular is an outstanding gap. DIT
and IDTxl have limited support for the partial information decomposition (PID) of
discrete random variables, but no other package also implements:

* the PID for **continuous, Gaussian** random variables,
* the **partial entropy decomposition** (PED),
* the **generalized information decomposition** (GID), or
* the **alpha-synergy decomposition**.

Several other classes of analysis included in Syntropy are likewise absent from
other primary packages, including spectral estimators for Gaussian
autoregressive processes and Lempel-Ziv estimators for discrete dynamical
processes.

The goal of Syntropy is to provide an accessible, easy-to-use API for scientists
who want to apply information-theoretic analyses to arbitrary datasets. It takes
considerable inspiration from `NetworkX <https://networkx.org/>`_: by providing a
high-level interface to complex analyses, Syntropy aims to lower the barrier to
entry for information theory so that these tools are accessible to scientists
from many different fields, without each researcher having to re-implement the
underlying measures.

To support this goal, Syntropy is built on a small, standard set of scientific
Python dependencies — ``numpy``, ``scipy``, and ``networkx`` — with the neural
(normalizing-flow) estimators available as an optional extra that additionally
requires ``torch`` and ``nflows`` (see :doc:`installation`).

Wherever possible, each information-theoretic function is provided with both
discrete and continuous implementations (e.g. a discrete *and* a continuous
entropy estimator, mutual information estimator, and so on). This lets
researchers choose the most appropriate tool for the data they have, rather than
being forced to transform their data to fit the estimator (for example, by
discretizing continuous data).

Syntropy is intended to be a "living package": the library of functions and
estimators will continue to grow as the field of multivariate information theory
develops.
