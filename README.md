# Syntropy

Syntropy is a Python package for information-theoretic analysis of discrete and continuous random variables. 

Wherever possible, syntropy provides discrete and Gaussian estimators for a given function (entropy, mutual information, total correlation, etc). 

#### Estimators 

##### Discrete 



##### Gaussian


#### The Distribution class

The Distribution class allows users to abstract away from the specific estimators (discrete or Gaussian) and defines a general "distribution" object which can be discrete or Gaussian and is equipped with the correct function estimators. 

```
from syntropy.distribution import Distribution
from syntropy.discrete.distributions import RANDOM_DIST_4

dist = Distribution(RANDOM_DIST_4)
```


#### Installation

Download the directory using ``git clone https://github.com/thosvarley/syntropy.git``

``cd`` into the directory and run ``pip install .`` (or ``pip install -e .`` for an editable install).
