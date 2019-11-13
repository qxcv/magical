#!/usr/bin/env python3
from setuptools import setup

setup(
    name='milbench',
    version='0.0.1',
    packages=['milbench'],
    install_requires=[
        # don't need TensorFlow yet (but will need an old TF copy when I do
        # imitation)
        # 'tensorflow>=1.13,<1.16',
        'pymunk~=5.6.0',
        'pyglet~=1.4.6',
        'Click~=7.0',
        'numpy~=1.17.4',
    ])
