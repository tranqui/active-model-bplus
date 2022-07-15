#!/usr/bin/env python3

from setuptools import setup, Extension, find_packages
import subprocess

try:
    version = subprocess.check_output(["git", "describe", "--tags"]).decode('ascii').strip()
    if '-' in version:
        version, ncommits, current_commit = version.split('-')
        version = '%s.dev%s' % (version, ncommits)
except: version = 'unknown-version'

with open("README.md", "r") as f:
    long_description = f.read()

required_modules = ['numpy', 'scipy', 'sympy', 'cloudpickle', 'functools', 'pathlib']
requirements = []
for module in required_modules:
    try: exec("import %s" % module)
    except: requirements += [module]

setup(
    name='activemodelbplus',
    version=version,
    license='GNU General Public License v3.0',

    author='Joshua F. Robinson',
    author_email='joshuarrr@protonmail.com',

    url='https://github.com/tranqui/active-model-bplus',
    description='Active Model B+ field theory: simple calculations for droplets and bulk coexistence',
    long_description=long_description,
    long_description_content_type="text/markdown",

    python_requires='>=3',
    ext_modules=[],
    install_requires=requirements,
    packages=find_packages('.'),

    classifiers=[
        "Programming Language :: Python :: 3",
    ],
 )
