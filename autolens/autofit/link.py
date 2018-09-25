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
    logger.debug(ex)


def path_for(path):
    """
    Generate a path in the ~/.autolens directory by taking the provided path, base64 encoding it and extracting the
    first and last five characters.

    Parameters
    ----------
    path: str
        The path where multinest output is apparently saved

    Returns
    -------
    actual_path: str
        The path where multinest output is actually saved
    """
    start = int(SUB_PATH_LENGTH / 2)
    end = SUB_PATH_LENGTH - start
    b64_string = b64encode(bytes(path, encoding="utf-8")).decode("utf-8")
    return "{}/{}".format(autolens_dir, b64_string[:start] + b64_string[-end:])


def make_linked_folder(path):
    """
    Create a folder in the ~/.autolens directory and create a sym link to it at the provided path.

    If both folders already exist then nothing is changed. If the source folder exists but the destination folder does
    not then the source folder is removed and replaced so as to conform to the behaviour that the user would expect
    should they delete the sym linked folder.

    Parameters
    ----------
    path: str
        The path where multinest output is apparently saved

    Returns
    -------
    actual_path: str
        The path where multinest output is actually saved
    """
    actual_path = path_for(path)
    if os.path.exists(actual_path) and not os.path.exists(path):
        shutil.rmtree(actual_path)
    try:
        os.mkdir(actual_path)
    except FileExistsError as e:
        logger.debug(e)
    try:
        os.symlink(actual_path, path)
    except FileExistsError as e:
        logger.debug(e)
    return actual_path
