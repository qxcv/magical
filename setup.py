#!/usr/bin/env python3
from setuptools import setup

setup(
    name='milbench',
    version='0.0.1',
    packages=['milbench'],
    install_requires=[
        'pymunk~=5.6.0',
        'pyglet~=1.5.0',
        'gym>=0.15.0,<0.18',
        'Click>=7.0',
        'numpy>=1.17.4',
        'cloudpickle>=1.2.2',
        'statsmodels>=0.10.2',
    ],
    extras_require={
        # for imitation baselines
        'baselines': [
            ('imitation @ '
             'git+https://github.com/HumanCompatibleAI/imitation.git'
             '#43e23fa0386d37b532ef58baae19ec67852ae8e4'),
        ],
    })
