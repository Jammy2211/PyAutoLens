#!/usr/bin/env bash

echo "__version__ = "`git branch | grep \* | cut -d ' ' -f2 | cut -d '/' -f2` > autolens/__init__.py

python setup.py sdist bdist_wheel
twine upload dist/* --skip-existing --username $PYPI_USERNAME --password $PYPI_PASSWORD

docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD
docker build -t rhayes777/autolens .
docker push rhayes777/autolens:latest
