import os
import logging
from os.path import expanduser
from base64 import b64encode

SUB_DIR_LENGTH = 5
AUTOLENS_FOLDER = ".autolens"

logger = logging.getLogger(__file__)

home = expanduser("~")
autolens_dir = "{}/{}".format(home, AUTOLENS_FOLDER)

try:
    os.mkdir(autolens_dir)
except Exception as e:
    logger.exception(e)


def directory_for(directory):
    return "{}/{}".format(autolens_dir, b64encode(bytes(directory, encoding="utf-8")).decode("utf-8")[:SUB_DIR_LENGTH])
