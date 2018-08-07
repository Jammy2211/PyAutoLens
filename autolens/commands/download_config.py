"""Reset Config."""

from .base import Base
from autolens import conf
import os

current_directory = os.getcwd()


class DownloadConfig(Base):
    """Reset Config!"""

    def run(self):
        if input("Are you sure? This will reset the state of your config").lower() == 'y':
            conf.remove_config()
            conf.download_config()
