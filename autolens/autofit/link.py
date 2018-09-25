import logging
import os
import shutil
from base64 import b64encode
from os.path import expanduser

SUB_PATH_LENGTH = 10
AUTOLENS_FOLDER = ".autolens"

logger = logging.getLogger(__file__)

home = expanduser("~")
autolens_dir = "{}/{}".format(home, AUTOLENS_FOLDER)

try:
    os.mkdir(autolens_dir)
except FileExistsError as ex:
    logger.exception(ex)


def path_for(path):
    start = int(SUB_PATH_LENGTH / 2)
    end = SUB_PATH_LENGTH - start
    b64_string = b64encode(bytes(path, encoding="utf-8")).decode("utf-8")
    return "{}/{}".format(autolens_dir, b64_string[:start] + b64_string[-end:])


def make_linked_folder(path):
    actual_path = path_for(path)
    if os.path.exists(actual_path) and not os.path.exists(path):
        shutil.rmtree(actual_path)
    try:
        os.mkdir(actual_path)
    except FileExistsError as e:
        logger.exception(e)
    try:
        os.symlink(actual_path, path)
    except FileExistsError as e:
        logger.exception(e)
    return actual_path
