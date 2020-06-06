#!/usr/bin/env python3
from setuptools import setup

setup(
    name='magical',
    version='0.0.1',
    packages=['magical'],
    install_requires=[
        'pymunk~=5.6.0',
        'pyglet>=1.3.0,<1.4.0',
        'gym>=0.15.0,<0.16',
        'Click>=7.0',
        'numpy>=1.17.4',
        'cloudpickle>=1.2.2',
        'statsmodels>=0.10.2',
    ])
