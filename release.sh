#!/usr/bin/env bash

python setup.py sdist bdist_wheel
twine upload dist/* --skip-existing --username $PYPI_USERNAME --password $PYPI_PASSWORD

docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD
docker build -t rhayes777/autolens .
docker push rhayes777/autolens:latest
