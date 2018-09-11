from autolens.commands.base import Base, current_directory
from autolens import conf
from autolens import pipeline
import colorama


def color(text, fore):
    """
    Apply a color to some text.

    Parameters
    ----------
    text: str
        The original text
    fore: colorama.ansi.AnsiFore
        The color to be applied to the text

    Returns
    -------
    text: str
        Colored text
    """
    return "{}{}{}".format(fore, text, colorama.Fore.RESET)


def blue(text):
    """
    Make text blue
    """
    return color(text, colorama.Fore.BLUE)


def red(text):
    """
    Make text red
    """
    return color(text, colorama.Fore.RED)


class Pipeline(Base):

    def run(self):
        name = self.options['<name>']
        conf.instance = conf.Config(self.config_path, self.output_path)
        if self.options['--info']:
            tup = pipeline.pipeline_dict[name]
            print()
            pl = tup.make()
            print(red(name))
            print(tup.doc)
            print()
            print(red("Phases"))
            print("\n".join(["{}\n   {}".format(phase.__class__.__name__, blue(phase.doc)) for phase in pl.phases]))
            return
        if name is not None:
            if name not in pipeline.pipeline_dict:
                print("No pipeline called '{}' found".format(name))
                return
            self.run_pipeline(pipeline.pipeline_dict[name].make())

        print_pipelines()

    def run_pipeline(self, pl):
        from autolens.imaging import image as im
        pl.run(im.load(self.image_path, pixel_scale=self.pixel_scale))

    @property
    def image_path(self):
        """
        Get the relative or absolute path to the input image. If the path does not begin with '/' then the current
        working directory will be prepended.

        Returns
        -------
        str: path
            The path to the image folder or image.
        """
        image_path = self.options['--image']
        if image_path is None:
            print("Please specify the path to the masked_image folder")
            return
        if not image_path.startswith("/"):
            image_path = "{}/{}".format(current_directory, image_path)
        return image_path

    @property
    def pixel_scale(self):
        """
        Returns
        -------
        pixel_scale: float
            The size of a single pixel, in arc seconds, as input by the user
        """
        return float(self.options['--pixel-scale'])

    @property
    def config_path(self):
        """
        Returns
        -------
        config_path: str
            The path to the configuration folder. Defaults to 'config' in the current working directory.
        """
        if '--config' in self.options:
            config_path = self.options['--config']
        else:
            config_path = 'config'
        return config_path

    @property
    def output_path(self):
        """
        Returns
        -------
        output_path: str
            The path to the configuration folder. Defaults to 'output' in the current working directory.
        """
        output_path = self.options['--output']
        if not output_path.startswith("/"):
            output_path = "{}/{}".format(current_directory, output_path)
        return output_path


def print_pipelines():
    """
    Prints a list of available pipelines taken from the pipeline dictionary.
    """
    print("Available Pipelines:\n")
    print(
        "\n".join(
            ["{}\n  {}".format(key, blue(value.short_doc)) for
             key, value
             in
             pipeline.pipeline_dict.items()]))
