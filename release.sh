#!/usr/bin/env bash

set -e

VERSION=$1

git flow release start $VERSION

echo "__version__ = '"$VERSION"'" > $PACKAGE_NAME/__init__.py

git add $PACKAGE_NAME/__init__.py

set +e
git commit -m "Incremented version number"
set -e

python setup.py sdist bdist_wheel
twine upload dist/* --skip-existing --username $PYPI_USERNAME --password $PYPI_PASSWORD

docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD
docker build -t autolens/$PACKAGE_NAME .
docker push autolens/$PACKAGE_NAME:latest

git flow release finish $VERSION

git checkout master
git push
git checkout develop
git push
