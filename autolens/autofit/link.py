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


def make_linked_folder(sym_path):
    """
    Create a folder in the ~/.autolens directory and create a sym link to it at the provided path.

    If both folders already exist then nothing is changed. If the source folder exists but the destination folder does
    not then the source folder is removed and replaced so as to conform to the behaviour that the user would expect
    should they delete the sym linked folder.

    Parameters
    ----------
    sym_path: str
        The path where multinest output is apparently saved

    Returns
    -------
    actual_path: str
        The path where multinest output is actually saved
    """
    source_path = path_for(sym_path)
    if os.path.exists(source_path) and not os.path.exists(sym_path):
        logger.info("Source {} exists but target {} does not. Removing source.".format(source_path, sym_path))
        shutil.rmtree(source_path)
    try:
        logger.info("Making source {}".format(source_path))
        os.mkdir(source_path)
        logger.info("Success")
    except FileExistsError as e:
        logger.info("Source already existed")
        logger.debug(e)
    try:
        logger.info("Making linking from source {} to sym {}".format(source_path, sym_path))
        os.symlink(source_path, sym_path)
        logger.info("Success")
    except FileExistsError as e:
        logger.info("Sym already existed")
        logger.debug(e)
    return source_path
