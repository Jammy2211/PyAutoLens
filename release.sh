#!/usr/bin/env bash

rm -rf $p/dist
rm -rf $p/build

set -e

VERSION=$1

git flow release start $VERSION

cat $PACKAGE_NAME/__init__.py | grep -v __version__ > temp

cat temp > $PACKAGE_NAME/__init__.py
rm temp
echo "__version__ = '"$VERSION"'" >> $PACKAGE_NAME/__init__.py

git add $PACKAGE_NAME/__init__.py

set +e
git commit -m "Incremented version number"
set -e

python setup.py test_autoarray sdist bdist_wheel
twine upload dist/* --skip-existing --username $PYPI_USERNAME --password $PYPI_PASSWORD

git flow release finish $VERSION

git checkout master
git push
git checkout development
git push

rm -rf $p/dist
rm -rf $p/build
