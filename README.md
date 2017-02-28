# FRANKEN-Z: Flexible Regression over Associated Neighbors with Kernel dEnsity estimatioN for Redshifts (Z)

**Note: This code is still in active development but should now be science ready. Please contact the authors if you're interested in using any of the code/methods outlined here on your data.**

**FRANKEN-Z** combines **machine learning** and **Bayesian inference** (i.e. likelihood fitting) methods in order to derive **sparse but accurate** mappings (i.e. projections) from a **target set** of photometric **probability distribution functions (PDFs)** onto a **training set** of photometric PDFs. In addition to taking measurement errors into account, FRANKEN-Z also can compute these mappings in the presence of **missing data**.

Using these discrete data space mappings, FRANKEN-Z enables the user to subsequently derive corresponding PDFs to any desired target space. Current applications are oriented towards deriving 1-D photometric redshifts, but possible future applications include deriving physical properties such as stellar mass, star formation rates, and metallicities.

Since it preserves object-level outputs, FRANKEN-Z can also be utilized as part of a broader Bayesian model. We outline some particular applications in a series of notebooks associated with the official release paper (Speagle et al. 2017).

This repository contains the core FRANKEN-Z code and associated release paper, along with the iPython notebooks used to conduct various tests. These notebooks (numbered) provide an overview of different aspects of the paper and code as well as relevant computational and statistical details.

This code is released under MIT License. Please cite the relevant papers/source if you use this code.

Notebook topics:
- how we generate mock data
- proper Bayesian population and hierarchical inference with redshift PDFs
- deriving photometric posteriors with missing data and selection effects
- hierarchical inference with "big data"
- tests on mock SDSS and HSC data
- cross-validation tests with real SDSS data

Files:
- **config/**: Configuration files for FRANKEN-Z.
- **data/**: Data used in the paper.
- **filters/**: Photometric filters.
- **paper/**: Current public draft of the release paper.
- **plots/**: Plots associated with the paper and notebooks.
- **seds/**: Galaxy spectral energy distribution (SED) templates.
- **frankenz.py**: Core set of FRANKEN-Z methods.
- **make_sim.py**: Set of methods for creating a quick mock catalog.

Contributors:
- Josh Speagle (Harvard)
- Boris Leistedt (NYU)
- Ben Hoyle (LMU Munich)

Related papers:
- *Hierarchical Bayesian Inference of Galaxy Redshift Distributions from Photometric Surveys* by Leistedt, Mortlock and Peiris. [arxiv:1602.05960](http://arxiv.org/abs/1602.05960).
- *Using Hierarchical Bayes and Machine Learning to Derive Photometric Redshifts from Observed Colors* by Speagle et al. (in preparation)
- *Validating Spectroscopic Contributions to Photometric Redshift Accuracy using HSC Survey Galaxy-Galaxy Lensing Data* by Speagle et al. (in preparation)
