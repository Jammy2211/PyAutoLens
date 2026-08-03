import os
from setuptools import setup

# Source builds stamp a version ABOVE every date release. Intra-family deps
# carry `>=` floors, so a low dev stamp (the old "1.0.dev0") makes a local
# checkout unsatisfiable against its own siblings. Release builds always set
# VERSION explicitly, so this default is never published.
version = os.environ.get("VERSION", "9999.0.0.dev0")

setup(
    version=version,
)
