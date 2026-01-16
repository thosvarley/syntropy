Information Theory Primer
=========================

This document provides a conceptual introduction to the core ideas of information theory, with particular attention to the challenges of estimation and higher-order information measures. For formal treatments, see Shannon's original work or Cover & Thomas's *Elements of Information Theory*. Much of the content here is based on the review article `"Information Theory for Complex Systems Scientists" <https://arxiv.org/abs/2304.12482>`_.

Entropy: Quantifying Uncertainty
--------------------------------

Shannon entropy is the foundational quantity of information theory. It quantifies the uncertainty associated with a random variable—how "surprised" we should expect to be when we observe its outcome.

The Intuition
^^^^^^^^^^^^^

There are two complementary ways to think about entropy:

**Expected Surprise.** Rare events are surprising; common events are not. The "surprise" or "self-information" of observing an outcome :math:`x` with probability :math:`P(x)` is:

.. math::

   h(x) = -\log P(x)

If an outcome is certain (:math:`P(x) = 1`), there is no surprise (:math:`h(x) = 0`). If an outcome is very rare, observing it is highly surprising. Entropy is simply the *expected* surprise across all possible outcomes:

.. math::

   H(X) = \mathbb{E}_X[h(x)] = -\sum_{x \in \mathcal{X}} P(x) \log P(x)

**Minimum Questions.** Entropy can also be interpreted as the minimum average number of yes/no questions needed to identify the state of a random variable. A fair 6-sided die requires about 2.58 questions on average, while a biased die (where one face appears most of the time) requires fewer questions—you can usually guess the common outcome first.

Key Properties
^^^^^^^^^^^^^^

- **Non-negativity:** :math:`H(X) \geq 0` (for discrete variables).
- **Maximum at uniformity:** Entropy is maximized when all outcomes are equally probable.
- **Additivity for independent variables:** If :math:`X_1` and :math:`X_2` are independent, then :math:`H(X_1, X_2) = H(X_1) + H(X_2)`.
- **Subadditivity:** In general, :math:`H(X_1, X_2) \leq H(X_1) + H(X_2)`, with equality only when the variables are independent.

Mutual Information: Quantifying Dependence
------------------------------------------

While entropy measures uncertainty about a single variable, mutual information measures the statistical dependence between two variables—how much knowing one reduces uncertainty about the other.

Definition and Interpretations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mutual information can be expressed in several equivalent ways:

.. math::

   I(X; Y) &= H(X) - H(X|Y) \\
           &= H(Y) - H(Y|X) \\
           &= H(X) + H(Y) - H(X, Y)

The first form says: mutual information is how much the entropy of :math:`X` decreases when we learn :math:`Y`. The third form shows it as the "overlap" between the individual entropies, minus the joint entropy.

**Bayesian Interpretation.** Mutual information equals the Kullback-Leibler divergence between the true joint distribution and the product of marginals:

.. math::

   I(X; Y) = D_{KL}(P(X,Y) \| P(X) P(Y))

This measures how much information we gain when updating from an assumption of independence to the actual joint distribution.

Local Mutual Information
^^^^^^^^^^^^^^^^^^^^^^^^

The *pointwise* or *local* mutual information for specific observations :math:`x` and :math:`y` is:

.. math::

   i(x; y) = \log \frac{P(x, y)}{P(x) P(y)} = \log \frac{P(x|y)}{P(x)}

Unlike the expected value, local mutual information can be negative—indicating that observing :math:`y` makes :math:`x` *less* likely than the marginal would suggest. This is sometimes called "misinformation."

Conditional Mutual Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conditional mutual information measures the dependence between :math:`X` and :math:`Y` given knowledge of a third variable :math:`Z`:

.. math::

   I(X; Y | Z) = H(X|Z) - H(X|Y, Z)

This is essential for identifying direct versus indirect dependencies in complex systems.

The Estimation Problem
----------------------

A central challenge in applied information theory is *estimation*: we rarely have access to the true probability distribution and must estimate it from finite samples.

Discrete Data
^^^^^^^^^^^^^

For discrete random variables, the naive approach is to estimate probabilities from observed frequencies (the "plug-in" estimator). However, this systematically **underestimates** the true entropy, especially when sample sizes are small relative to the number of possible states.

The problem is that rare events may not appear in a finite sample, leading us to assign them zero probability. Several bias-correction methods exist (Miller-Madow, Grassberger, Bayesian estimators), but the fundamental issue remains: estimating entropy from discrete data requires careful attention to sample size and the number of possible states.

Syntropy's ``syntropy.discrete`` module computes exact information measures from fully specified probability distributions. If you only have samples, you must first estimate the distribution.

Continuous Data
^^^^^^^^^^^^^^^

Continuous random variables present additional challenges. The direct analog of Shannon entropy is *differential entropy*:

.. math::

   h(X) = -\int p(x) \log p(x) \, dx

Unlike discrete entropy, differential entropy can be negative and depends on the coordinate system. Several strategies exist for estimating information from continuous data:

**Binning (Coarse-Graining).** Discretize the continuous data into bins, then apply discrete entropy formulas. This is simple but problematic: results depend heavily on bin width, and the "true" differential entropy is only recovered in the limit of infinitely fine bins with infinite data.

**Gaussian Assumption.** If the data are (approximately) Gaussian, entropy depends only on the covariance matrix:

.. math::

   H(X) = \frac{1}{2} \log \left( (2 \pi e)^d |\Sigma| \right)

where :math:`d` is the dimensionality and :math:`|\Sigma|` is the determinant of the covariance matrix. This yields closed-form solutions and is computationally efficient, but obviously requires the Gaussianity assumption to hold. Syntropy's ``syntropy.gaussian`` module implements these estimators.

**K-Nearest Neighbor (KNN) Estimators.** KNN estimators (e.g., Kraskov-Stogbauer-Grassberger) avoid explicit density estimation by using the distances to nearby points to estimate local density. They are:

- Non-parametric (no distributional assumptions)
- Adaptive to local data structure
- Effective across varying dimensions

However, they can be computationally expensive for large datasets and sensitive to the choice of :math:`k`. Syntropy's ``syntropy.knn`` module implements KNN-based estimators.

**Neural Estimators.** Modern approaches use neural networks (e.g., normalizing flows) to estimate densities or directly estimate information-theoretic quantities. These can be powerful for high-dimensional data but require careful training and more computational resources. Syntropy's ``syntropy.neural`` module implements flow-based estimators.

Choosing an Estimator
^^^^^^^^^^^^^^^^^^^^^

There is no universally "best" estimator. The choice depends on:

- **Data type:** Discrete data requires discrete estimators; continuous data requires continuous estimators.
- **Sample size:** KNN and neural estimators generally need more samples than parametric methods.
- **Distributional assumptions:** If your data are approximately Gaussian, Gaussian estimators will be more efficient. If not, they will be biased.
- **Dimensionality:** High-dimensional estimation is inherently difficult. KNN estimators suffer from the curse of dimensionality, though less severely than binning approaches.
- **Computational budget:** Gaussian estimators are fast; neural estimators are slow.

Higher-Order Information
------------------------

Mutual information captures pairwise dependencies, but many complex systems exhibit irreducibly higher-order interactions—statistical dependencies that cannot be reduced to pairwise relationships.

Total Correlation
^^^^^^^^^^^^^^^^^

Total correlation (also called multi-information) generalizes mutual information to :math:`N` variables:

.. math::

   TC(X_1, \ldots, X_N) = \sum_{i=1}^N H(X_i) - H(X_1, \ldots, X_N)

This measures the total amount of statistical dependence in the system—how much the joint entropy falls short of the sum of individual entropies.

Dual Total Correlation
^^^^^^^^^^^^^^^^^^^^^^

Dual total correlation (also called binding information) takes a complementary perspective:

.. math::

   DTC(X_1, \ldots, X_N) = H(X_1, \ldots, X_N) - \sum_{i=1}^N H(X_i | X_{-i})

where :math:`X_{-i}` denotes all variables except :math:`X_i`. This measures the information that is *redundantly* encoded across variables.

O-Information
^^^^^^^^^^^^^

The O-information combines total correlation and dual total correlation:

.. math::

   \Omega(X_1, \ldots, X_N) = TC(X_1, \ldots, X_N) - DTC(X_1, \ldots, X_N)

The sign of O-information indicates whether a system is dominated by redundancy (:math:`\Omega > 0`) or synergy (:math:`\Omega < 0`):

- **Redundancy-dominated:** Variables carry overlapping, duplicated information about each other.
- **Synergy-dominated:** Information emerges from the combination of variables that cannot be found in subsets.

S-Information
^^^^^^^^^^^^^

The S-information is the sum of total correlation and dual total correlation:

.. math::

   \Sigma(X_1, \ldots, X_N) = TC(X_1, \ldots, X_N) + DTC(X_1, \ldots, X_N)

This provides an overall measure of statistical structure in the system.

Partial Information Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Partial Information Decomposition (PID) attempts to decompose the mutual information between a set of sources and a target into distinct components:

- **Redundancy:** Information provided identically by multiple sources.
- **Unique information:** Information provided by only one source.
- **Synergy:** Information available only when sources are considered jointly.

For two sources :math:`X_1, X_2` and target :math:`Y`:

.. math::

   I(X_1, X_2; Y) = \text{Red}(X_1, X_2 \to Y) + \text{Unq}(X_1 \to Y) + \text{Unq}(X_2 \to Y) + \text{Syn}(X_1, X_2 \to Y)

The XOR function is the canonical example of synergy: neither input alone predicts the output, but together they determine it completely.

Challenges in Higher-Order Information
--------------------------------------

Higher-order information theory is an active research area with several fundamental challenges:

Definitional Ambiguity
^^^^^^^^^^^^^^^^^^^^^^

Unlike entropy and mutual information, there is no unique, universally accepted definition of redundancy and synergy. Multiple proposals exist (I_min, I_BROJA, I_CCS, I_sx, etc.), each satisfying different axioms and yielding different decompositions for the same data. The "correct" measure may depend on the specific scientific question.

Estimation Difficulty
^^^^^^^^^^^^^^^^^^^^^

Higher-order measures compound the estimation challenges of basic information theory:

- **Sample complexity:** Estimating joint distributions over many variables requires exponentially more data.
- **Bias amplification:** Biases in entropy estimation propagate and amplify through higher-order formulas.
- **Dimensionality:** KNN and density-based estimators struggle in high dimensions.

Computational Complexity
^^^^^^^^^^^^^^^^^^^^^^^^

The number of possible higher-order interactions grows combinatorially. For :math:`N` variables, there are :math:`\binom{N}{k}` possible :math:`k`-th order interactions. Exhaustively computing all higher-order terms quickly becomes infeasible.

Interpretation Challenges
^^^^^^^^^^^^^^^^^^^^^^^^^

Even when higher-order measures can be computed, interpreting them is subtle:

- Negative O-information indicates synergy dominance, but doesn't localize *where* the synergy occurs.
- PID decomposes information, but the meaning of "redundancy" remains debated.
- Different normalization choices can lead to different conclusions about the same system.

Practical Recommendations
-------------------------

1. **Start simple.** Compute pairwise mutual information before moving to higher-order measures. Many questions can be answered without the complexity of multivariate methods.

2. **Validate with known distributions.** Test your analysis pipeline on synthetic data with known information-theoretic properties before applying to real data.

3. **Consider multiple estimators.** If results depend heavily on the choice of estimator, treat conclusions with caution.

4. **Mind your sample size.** Information estimation is data-hungry. Underpowered analyses can yield unreliable or misleading results.

5. **Report uncertainty.** Use bootstrap or other resampling methods to quantify estimation uncertainty.

6. **Be skeptical of high-dimensional claims.** Estimating information in high dimensions is inherently difficult. Extraordinary claims require extraordinary evidence.

Further Reading
---------------

- Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*.
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
- Varley, T. F. (2024). Information Theory for Complex Systems Scientists. *arXiv:2304.12482*.
- Timme, N. M., & Lapish, C. (2018). A Tutorial for Information Theory in Neuroscience. *eNeuro*.
- Mediano, P. A. M., et al. (2022). Greater than the parts: A review of the information decomposition approach to causal emergence. *Philosophical Transactions of the Royal Society A*.
