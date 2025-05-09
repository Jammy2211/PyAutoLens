import os
from codecs import open
from os import environ
from os.path import abspath, dirname, join

from setuptools import setup

this_dir = abspath(dirname(__file__))
with open(join(this_dir, "README.rst"), encoding="utf-8") as file:
    long_description = file.read()

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")

version = environ.get("VERSION", "1.0.dev0")
requirements.extend(
    [
        f"autoarray=={version}",
        f"autoconf=={version}",
        f"autofit=={version}",
        f"autogalaxy=={version}",
    ]
)

def config_packages(directory):
    paths = [directory.replace("/", ".")]
    for path, directories, filenames in os.walk(directory):
        for directory in directories:
            paths.append(f"{path}/{directory}".replace("/", "."))
    return paths

setup(
    version=version,
    install_requires=requirements,
)
