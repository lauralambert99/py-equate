# -*- coding: utf-8 -*-
"""
Created on Thu May  1 08:01:36 2025

@author: laycocla
"""

from setuptools import setup, find_packages

setup(
    name='pyequate',
    version='0.1.0',
    packages = find_packages(),
    description="Score equating methods for testing",
    authors=['Laura Lambert', 'Yu Bao'],
    author_email=['laycocla@jmu.edu', 'bao2yx@jmu.edu'],
    license='GPL-3.0',
    packages=['pyequate'],
    install_requires=['numpy', 'pandas', 'scipy', 'statsmodels']
)
