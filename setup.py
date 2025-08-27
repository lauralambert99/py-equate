# -*- coding: utf-8 -*-
"""
Created on Thu May  1 08:01:36 2025

@author: laycocla
"""

from setuptools import setup, find_packages

setup(
    name='equate',
    version='0.1.0',
    packages = find_packages(),
    description="Score equating methods for testing",
    authors=['Laura Lambert', 'Yu Bao'],
    install_requires=['numpy', 'pandas', 'scipy', 'statsmodels']
)
