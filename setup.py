#!/usr/bin/env python3
import os

from setuptools import find_packages, setup

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def readme():
    """Load README for use as package's long description."""
    with open(os.path.join(THIS_DIR, 'README.md'), 'r') as fp:
        return fp.read()


def get_version():
    locals_dict = {}
    with open(os.path.join(THIS_DIR, 'magical', 'version.py'), 'r') as fp:
        exec(fp.read(), globals(), locals_dict)
    return locals_dict['__version__']


setup(name='magical-il',
      version=get_version(),
      author='Sam Toyer',
      license='ISC',
      url='https://github.com/qxcv/magical/',
      description='MAGICAL is a benchmark suite for robust imitation learning',
      python_requires='>=3.6',
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
          'opencv-python-headless==4.*',
      ],
      extras_require={
          'dev': [
              'pytest~=6.1.2',
              'pytest-xdist~=2.1.0',
              'isort~=5.0',
              'yapf~=0.30.0',
              'flake8~=3.8.3',
              'autoflake~=1.4',
              'pytest-flake8~=1.0.6',
              'pytest-isort~=1.2.0',
          ],
      })
