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

    
setup(
    name="frankenz",
    url="https://github.com/joshspeagle/frankenz",
    version="0.0.1",
    author="Josh Speagle",
    author_email="jspeagle@cfa.harvard.edu",
    packages=["frankenz"],
    license="LICENSE",
    description="photo-z's with hierarchical bayes and machine learning",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "matplotlib", "six"],
)
