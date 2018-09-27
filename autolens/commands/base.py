"""The base command."""

import os
from functools import wraps

current_directory = os.getcwd()


def prepend_working_directory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        simple_path = func(*args, **kwargs)
        if not simple_path.startswith("/"):
            simple_path = "{}/{}".format(current_directory, simple_path)
        return simple_path

    return wrapper


class Base(object):
    """A base command."""

    def __init__(self, options, *args, **kwargs):
        self.options = options
        self.args = args
        self.kwargs = kwargs

    def run(self):
        raise NotImplementedError('You must implement the run() method yourself!')
