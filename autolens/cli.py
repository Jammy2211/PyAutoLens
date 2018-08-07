"""
autolens

Usage:
  autolens reset_config
  autolens pipeline
  autolens -h | --help
  autolens --version

Options:
  -h --help                         Show this screen.
  --version                         Show version.

Examples:
  autolens reset_config
  autolens pipeline

Help:
  For help using this tool, please open an issue on the Github repository:
  https://github.com/Jammy2211/PyAutoLens
"""

from inspect import getmembers, isclass

from docopt import docopt

from . import __version__


def main():
    """Main CLI entrypoint."""
    import autolens.commands

    options = docopt(__doc__, version=__version__)

    # Here we'll try to dynamically match the command the user is trying to run
    # with a pre-defined command class we've already created.
    for (k, v) in options.items():
        if hasattr(autolens.commands, k) and v:
            module = getattr(autolens.commands, k)
            command = [command[1] for command in getmembers(module, isclass) if command[0] != 'Base'][0]
            command = command(options)
            command.run()


if __name__ == "__main__":
    main()
