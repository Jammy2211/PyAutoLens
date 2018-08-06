"""The hello command."""

from .base import Base, await_input


class ResetConfig(Base):
    """Say hello, world!"""

    def run(self):
        print("Are you sure? This will reset the state of your config")
        if await_input().lower() == 'y':
            print("REMOVING")
