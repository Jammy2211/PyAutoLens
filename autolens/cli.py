"""
autolens

Usage:
  autolens reset_config
  autolens pipeline
  autolens pipeline <name> --info
  autolens pipeline <name> (--image=<image>) (--pixel-scale=<pixel-scale>) [--config=<config>] [--output=<output>]
  autolens -h | --help
  autolens --version

Options:
  -h --help                         Show this screen.
  --version                         Show version.
  --pixel-scale=<pixel-scale>       The scale of one pixel in the image.
  --image=<image>                   The path to the folder that contains the image data.
  --config=<config>                 The path to the folder that contains the configuration [default: config].
  --output=<output>                 The path to the folder where data should be output [default: output].

Examples:
  autolens reset_config
  autolens pipeline
  autolens pipeline profile --image=hst_0/ --pixel-scale=0.05
  autolens pipeline profile --image=hst_0 --pixel-scale=0.05 --output=output_folder --config=config_folder

Help:
  For help using this tool, please open an issue on the Github repository:
  https://github.com/Jammy2211/PyAutoLens
"""

from inspect import getmembers, isclass

from docopt import docopt

from . import __version__


def main():
    """Main CLI entry point."""
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
