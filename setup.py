#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'frankenz', '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)

try:
    import pypandoc
    with open('README.md', 'r') as f:
        txt = f.read()
    txt = re.sub('<[^<]+>', '', txt)
    long_description = pypandoc.convert(txt, 'rst', 'md')
except ImportError:
    long_description = open('README.md').read()
    
setup(
    name="frankenz",
    url="https://github.com/joshspeagle/frankenz",
    version=__version__,
    author="Joshua S Speagle",
    author_email="jspeagle@cfa.harvard.edu",
    packages=["frankenz"],
    license="MIT",
    description="A photometric redshift monstrosity",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE", "AUTHORS.md"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "matplotlib", "six", "pandas"],
    keywords=["photo-z", "photometric redshift", "bayesian",
              "template fitting", "machine learning", "nearest neighbors",
              "self-organizing map", "growing neural gas"],
    classifiers=["Development Status :: 3 - Alpha",
                 "License :: OSI Approved :: MIT License",
                 "Natural Language :: English",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.6",
                 "Operating System :: OS Independent",
                 "Topic :: Scientific/Engineering :: Astronomy",
                 "Intended Audience :: Science/Research"]
)
