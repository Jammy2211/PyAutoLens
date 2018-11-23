#!/usr/bin/env bash

set -e

BRANCH=`git branch | grep \* | cut -d ' ' -f2`
BRANCH_TYPE=`echo $BRANCH | cut -d '/' -f1`
VERSION=`echo $BRANCH | cut -d '/' -f2`

if [ $BRANCH_TYPE != "release" ]  
then
	echo "Must be on a release branch. Type git-flow release start a new release."
	exit 1
fi

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
