#!/usr/bin/env python

from setuptools import setup, find_packages
import io
import os

here = os.path.abspath(os.path.dirname(__file__))

NAME = 'raocp'

# Import version from file
version_file = open(os.path.join(here, 'VERSION'))
VERSION = version_file.read().strip()

DESCRIPTION = 'Solver for multistage risk-averse optimal control problems'


# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=['Ruairi Moran', 'Zichi Zhang', 'Pantelis Sopasakis'],
      author_email='anonymous@qub.ac.uk',
      license='',
      packages=find_packages(
          exclude=["tests"]),
      include_package_data=True,
      install_requires=[
          'numpy', 'scipy', 'PythonTurtle'
      ],
      classifiers=[
          'Programming Language :: Python'
      ],
      keywords=['wind'],
      url=(
          'https://github.com/smokinmirror/rampc-toolbox'
      ),
      zip_safe=False)
