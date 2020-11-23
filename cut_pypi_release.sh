#!/bin/bash

# Upload a new release to PyPI

set -e

# make sure git directory is clean
# (see https://stackoverflow.com/a/3879077)
if [ -z "$(git status -s)" ]; then
    echo "git index clean, ready to upload"
else
    echo "!! could not confirm that git index is clean !!"
    echo "check git status & commit uncommitted changes before continuing"
    exit 1
fi

# remove old dist files
rm -rf dist/*
# now build & upload
python setup.py sdist bdist_wheel
python -m twine upload --repository pypi dist/*
