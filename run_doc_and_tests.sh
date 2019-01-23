#!/bin/bash


# generate the docs and open them in firefox
cd docs
make html

# run the tests
cd ..
#py.test -s
py.test
