#!/bin/bash -x

## Clean-up script (removes build artifacts etc)

rm -fr build
find . -name \*.so -exec rm {} +
find . -name __pycache__ -exec rm -r {} +
rm -rf *.egg-info
rm -rf .pytest_cache
git clean -dfX
