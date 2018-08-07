"""Reset Config."""

from .base import Base, current_directory
from autolens import conf


class DownloadConfig(Base):
    """Reset Config!"""

    def run(self):
        if conf.is_config(current_directory):
            if input("Are you sure? This will reset the state of your config. (y/n)\n").lower() != 'y':
                return
            conf.remove_config(current_directory)
        conf.download_config(current_directory)
