#!/usr/bin/env bash

rm -rf dist
rm -rf build

set -e

VERSION=$1

git flow release start $VERSION

echo "__version__ = '"$VERSION"'" > $PACKAGE_NAME/__init__.py

git add $PACKAGE_NAME/__init__.py

set +e
git commit -m "Incremented version number"
set -e

python setup.py test
python setup.py sdist bdist_wheel

twine upload dist/* --skip-existing --username $PYPI_USERNAME --password $PYPI_PASSWORD

# sudo docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD
# sudo docker build -t autolens/$PACKAGE_NAME .
# sudo docker push autolens/$PACKAGE_NAME:latest

git flow release finish $VERSION

git checkout master
git push
git checkout development
git push

