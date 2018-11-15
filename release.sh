#!/usr/bin/env bash

set -e

BRANCH=`git branch | grep \* | cut -d ' ' -f2`
BRANCH_TYPE=`echo $BRANCH | cut -d ' ' -f1`
VERSION=`echo $BRANCH | cut -d '/' -f2`

if [ $BRANCH_TYPE!="release" ]  
then
	echo "Must be on a release branch. Type git-flow release start a new release."
	exit 1
fi

echo "__version__ = "`git branch | grep \* | cut -d ' ' -f2 | cut -d '/' -f2` > autolens/__init__.py

python setup.py sdist bdist_wheel
twine upload dist/* --skip-existing --username $PYPI_USERNAME --password $PYPI_PASSWORD

docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD
docker build -t rhayes777/autolens .
docker push rhayes777/autolens:latest
