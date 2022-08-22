#!/usr/bin/python3

from setuptools import setup
from Cython.Build import cythonize

with open('README.md') as f:
    readme = f.read()

meta = {}
with open("peakTree/_meta.py") as fp:
    exec(fp.read(), meta)

setup(
    name='peakTree',
    version=meta['__version__'],
    description='',
    long_description=readme,
    long_description_content_type='text/markdown',
    author=meta['__author__'],
    author_email='radenz@tropos.de',
    url='https://github.com/martin-rdz/peakTree',
    download_url='',
    license='MIT License',
    packages=['peakTree'],
    include_package_data=True,
    python_requires='>=3.8',
    # automatic installation of the dependencies did not work with the test.pypi
    # below the try to fix it
    setup_requires=['wheel', 'numpy==1.22.0', 'scipy>=1.6', 'netCDF4>=1.4.2', 
                      'matplotlib>=2.2.2', 'toml>=0.10.0', 'numba>=0.45.1', 'graphviz', 'loess'],
    install_requires=['wheel', 'numpy==1.22.0', 'scipy>=1.6', 'netCDF4>=1.4.2', 
                      'matplotlib>=2.2.2', 'toml>=0.10.0', 'numba>=0.45.1', 'graphviz', 'loess'],
    #ext_modules=cythonize("pyLARDA/peakTree_fastbuilder.pyx"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
