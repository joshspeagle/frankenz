frankenz
=========
#### A photometric redshift monstrosity.

**WARNING: This project is under active development and not yet stable.**

`frankenz` is a Pure Python implementation of a variety of methods to quickly
yet robustly perform (hierarchical) Bayesian inference using large
(but discrete) sets of (possibly noisy) models with (noisy) photometric data.
The code also contains a number of additional utilities, including:
- a module for generating quick mocks (along with filter curves and SEDs), 
- several manifold-learning algorithms,
- a flexible set of photometric likelihoods,
- fast kernel density estimation, and
- PDF-oriented plotting utilities.

Paper forthcoming.

### Documentation
**Currently nonexistent.** See the demos for examples.

### Installation
`frankenz` can be installed via
```
pip install frankenz
```
Alternately, it can also be installed by running
```
python setup.py install
```
from inside the repository.

### Demos
Several Jupyter notebooks that demonstrate most of the available features
can be found [here](https://github.com/joshspeagle/frankenz/tree/master/demos).
