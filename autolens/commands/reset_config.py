"""Reset Config."""

from .base import Base
from autolens import config


class ResetConfig(Base):
    """Reset Config!"""

    def run(self):
        if input("Are you sure? This will reset the state of your config").lower() == 'y':
            config.remove_config()
            config.download_config()
