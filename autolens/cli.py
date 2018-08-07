"""
autolens

Usage:
  autolens download_config
  autolens pipeline
  autolens pipeline <name> [--pixel-scale=<pixel-scale>]
  autolens -h | --help
  autolens --version

Options:
  -h --help                         Show this screen.
  --version                         Show version.
  --image                           Specify the image path.
  --config                          Specify the config path.
  --pixel-scale=<pixel-scale>       The scale of one pixel in the image [default: 0.1].

Examples:
  autolens download_config
  autolens pipeline

Help:
  For help using this tool, please open an issue on the Github repository:
  https://github.com/Jammy2211/PyAutoLens
"""

from inspect import getmembers, isclass

from docopt import docopt
from exc import CLIException

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
            try:
                command.run()
            except CLIException as e:
                print(e.args[0])


if __name__ == "__main__":
    main()
