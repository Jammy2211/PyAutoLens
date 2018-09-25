import os
import logging
from os.path import expanduser

logger = logging.getLogger(__file__)

home = expanduser("~")
autolens_dir = "{}/.autolens".format(home)

try:
    os.mkdir(autolens_dir)
except Exception as e:
    logger.exception(e)
