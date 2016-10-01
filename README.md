# FRANKEN-Z

This repository eventually will contain code to (1) compute photometric redshifts using a combination of machine learning and hierarchical bayesian modeling and (2) generate mock catalogs to conduct simple tests.

This code is released under MIT License. Please cite the relevant papers/source if you use this code.

Content:
- *mock_survey.ipynb*: Notebook to generate a photometric survey mock with realistic fluxes/errors, redshifts, and underlying galaxy types based on the priors provided in [BPZ](http://www.stsci.edu/~dcoe/BPZ/). Draws heavily from a similar notebook written by [@ixkael](https://github.com/ixkael/Photoz-tools).
- *filters* and *seds* contain a variety of galaxy SED templates and photometric filters.

Contributors:
- Josh Speagle (Harvard)
- Boris Leistedt (NYU)
- Additional people...

Related papers:
- *Hierarchical Bayesian inference of galaxy redshift distributions from photometric surveys* by Leistedt, Mortlock and Peiris. [arxiv:1602.05960](http://arxiv.org/abs/1602.05960).

