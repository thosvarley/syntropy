# Syntropy

Syntropy is a Python package for information-theoretic analysis of discrete and continuous random variables. 

Wherever possible, syntropy provides discrete and Gaussian estimators for a given function (entropy, mutual information, total correlation, etc). 

#### Estimators 

Information theoretic analysis requires estimating entropies from different kinds of data - discrete and continuous. This is a non-trivial problem, especially in the case of real-valued data, as the underlying probability distributions can be non-parametric. The Syntropy package implements functions with three classes of estimators: discrete, continuous using Gaussian assumptions, and continuous estimators using normalizing flows.  

##### Discrete 

The discrete estimators are built on discrete probability distributions and the measures computed directly:

$$H(X) = -\sum_{x\in\mathcal{X}}P_X(x)\log P_X(x)$$

The distirbutions are represented in Python as dictionaries where the keys are tuples representing the joint-state of every element, and the values are the probabilities of that state. For example, an XOR distribution would look like:

```python
xor_distribution = {
    (0,0,0) : 1/4,
    (0,1,1) : 1/4,
    (1,0,1) : 1/4,
    (1,1,0) : 1/4,
}
```
The discrete estimators typically take in the distribution, and the indices of the relevant variables, represented as tuples. For example, the mutual information estimator syntax is:

```python
_, mi = mutual_information((0,1), (2,), xor_distribution) # Returns 1 bit.
```

Syntropy also includes a number of utility functions for manipulating discrete probability distributions.

##### Gaussian

Gaussian estimators assume that real valued data are drawn from a $k$-dimensional normal distribution $P_{X}(x)$ parameterized by some covariance matrix $\Sigma_{X}$. The entropy of that distribution is given by:


$$H(X) = \frac{k}{2}\log(2\pi\textnormal{e}) + \frac{1}{2}\log |\Sigma_{X}|$$

Where $|\Sigma_{X}|$ is the determinant of the covariance matrix. 

##### Neural 


#### Installation

Download the directory using ``git clone https://github.com/thosvarley/syntropy.git``

``cd`` into the directory and run ``pip install .`` (or ``pip install -e .`` for an editable install).

##### Unit testing 
If you have ``pytest`` installed, you can double-check the unit tests with ``pytest tests/`` from the root directory. 
