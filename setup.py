#!/usr/bin/env python3
import os
from setuptools import find_packages, setup


def readme():
    """Load README for use as package's long description."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(this_dir, 'README.md'), 'r') as fp:
        return fp.read()


setup(
    name='magical-il',
    version='0.0.1alpha1',
    author='Sam Toyer',
    license='ISC',
    url='https://github.com/qxcv/magical/',
    description='MAGICAL is a benchmark suite for robust imitation learning',
    python_requires='>=3.7',
    long_description=readme(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'pymunk~=5.6.0',
        'pyglet==1.5.*',
        'gym==0.17.*',
        'Click>=7.0',
        'numpy>=1.17.4',
        'cloudpickle>=1.2.2',
        'statsmodels>=0.10.2',
        'requests>=2.20.0,==2.*',
    ])
