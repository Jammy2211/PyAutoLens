import os
from codecs import open
from os import environ
from os.path import abspath, dirname, join

from setuptools import find_packages, setup

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
    name="autolens",
    version=version,
    description="Open-Source Strong Lensing",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/Jammy2211/PyAutoLens",
    author="James Nightingale and Richard Hayes",
    author_email="james.w.nightingale@durham.ac.uk",
    include_package_data=True,
    license="MIT License",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="cli",
    packages=find_packages(exclude=["docs", "test_autolens", "test_autolens*"])
    + config_packages("autolens/config"),
    install_requires=requirements,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
