import os
from setuptools import setup

version = os.environ.get("VERSION", "1.0.dev0")

setup(
    version=version,
)
