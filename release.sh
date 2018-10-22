#!/usr/bin/env bash

python setup.py sdist bdist_wheel
twine upload dist/* --skip-existing --username $PYPI_USERNAME --password $PYPI_PASSWORD
