"""Reset Config."""

from autofit import conf
from .base import Base, current_directory

config_path = "{}/config".format(current_directory)


class ResetConfig(Base):
    """Reset Config!"""

    def run(self):
        if conf.is_config(config_path):
            if input("Are you sure? This will reset the state of your config. (y/n)\n").lower() != 'y':
                return
            conf.remove_config(config_path)
        conf.copy_default(config_path)
