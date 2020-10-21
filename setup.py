#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name='magical-il',
    version='0.0.1alpha0',
    packages=find_packages(),
    install_requires=[
        'pymunk~=5.6.0',
        'pyglet==1.5.*',
        'gym==0.17.*',
        'Click>=7.0',
        'numpy>=1.17.4',
        'cloudpickle>=1.2.2',
        'statsmodels>=0.10.2',
    ])
