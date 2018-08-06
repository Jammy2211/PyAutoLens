"""Reset Config."""

from .base import Base, await_input


class ResetConfig(Base):
    """Reset Config!"""

    def run(self):
        print("Are you sure? This will reset the state of your config")
        if await_input().lower() == 'y':
            print("REMOVING")
