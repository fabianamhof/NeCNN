#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages
from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'neat-python',
    'graphviz',
    'matplotlib',
    'pyqt5',
    'torch',
    'numpy'
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest>=3',
]

setup(
    name='NeCNN',
    install_requires=requirements,
    packages=find_packages(),
    version='0.1.0',
)
