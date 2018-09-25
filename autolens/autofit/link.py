import os
import logging
from os.path import expanduser
from base64 import b64encode

SUB_PATH_LENGTH = 5
AUTOLENS_FOLDER = ".autolens"

logger = logging.getLogger(__file__)

home = expanduser("~")
autolens_dir = "{}/{}".format(home, AUTOLENS_FOLDER)

try:
    os.mkdir(autolens_dir)
except FileExistsError as e:
    pass


def path_for(path):
    return "{}/{}".format(autolens_dir, b64encode(bytes(path, encoding="utf-8")).decode("utf-8")[:SUB_PATH_LENGTH])


def make_linked_file(path):
    actual_path = path_for(path)
    open(actual_path, 'a').close()
    try:
        os.symlink(actual_path, path)
    except FileExistsError:
        pass
    return actual_path
