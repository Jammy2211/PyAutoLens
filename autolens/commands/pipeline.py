import colorama

from autofit import conf
from autolens import runners
from autolens.commands.base import Base, prepend_working_directory


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
        try:
            if self.options['--info']:
                tup = runners.pipeline_dict[name]
                print()
                pl = tup.make()
                print(red(name))
                print(tup.doc)
                print()
                print(red("Phases"))
                print("\n".join(["{}\n   {}".format(phase.__class__.__name__, blue(phase.doc)) for phase in pl.phases]))
                return
            if name is not None:
                if name == "test":
                    self.run_pipeline(runners.TestPipeline())
                    return
                self.run_pipeline(runners.pipeline_dict[name].make())
                return
        except KeyError:
            print("Pipeline '{}' does not exist.\n".format(name))

        print_pipelines()

    def run_pipeline(self, pl):
        from autolens.data.imaging import image as im
        if self.is_using_hdu:
            image = im.load_imaging_from_fits(self.data_path, self.image_hdu, self.noise_hdu, self.psf_hdu,
                                              self.pixel_scale)
        else:
            image = im.load_imaging_from_path(self.image_path, self.noise_path, self.psf_path,
                                              pixel_scale=self.pixel_scale)
        pl.run(image)

    @property
    def is_using_hdu(self):
        """
        Returns
        -------
        is_using_hdu: bool
            True iff --datas option is set. --datas is the path to a file with multiple datas layers accessible by setting
            hdus.
        """
        return self.options["--datas"] is not None

    @property
    def image_hdu(self):
        """
        Returns
        -------
        str: image_hdu
            The hdu of the regular datas in the datas file
        """
        return int(self.options["--regular-hdu"])

    @property
    def noise_hdu(self):
        """
        Returns
        -------
        str: noise_hdu
            The hdu of the noise datas in the datas file
        """
        return int(self.options["--noise-hdu"])

    @property
    def psf_hdu(self):
        """
        Returns
        -------
        str: psf_hdu
            The hdu of the psf datas in the datas file
        """
        return int(self.options["--psf-hdu"])

    @property
    @prepend_working_directory
    def image_path(self):
        """
        Get the relative or absolute path to the input datas_. If the path does not begin with '/' then the current
        working directory will be prepended.

        Returns
        -------
        str: path
            The path to the regular
        """
        return self.options['--regular']

    @property
    @prepend_working_directory
    def data_path(self):
        """
        Get the relative or absolute path to the input datas. Input datas includes datas_, noise and psf with different
        hdu values input by the user. If the path does not begin with '/' then the current working directory will be
        prepended.

        Returns
        -------
        str: path
            The path to the datas
        """
        return self.options['--datas']

    @property
    @prepend_working_directory
    def noise_path(self):
        """
        Get the relative or absolute path to the input noise. If the path does not begin with '/' then the current
        working directory will be prepended.

        Returns
        -------
        str: path
            The path to the noise
        """
        return self.options['--noise']

    @property
    @prepend_working_directory
    def psf_path(self):
        """
        Get the relative or absolute path to the input psf. If the path does not begin with '/' then the current
        working directory will be prepended.

        Returns
        -------
        str: path
            The path to the psf folder or psf.
        """
        return self.options['--psf']

    @property
    def pixel_scale(self):
        """
        Returns
        -------
        pixel_scales: float
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
    @prepend_working_directory
    def output_path(self):
        """
        Returns
        -------
        output_path: str
            The path to the configuration folder. Defaults to 'output' in the current working directory.
        """
        return self.options['--output']


def print_pipelines():
    """
    Prints a list of available runners taken from the pipeline dictionary.
    """
    print("Available Pipelines:\n")
    print(
        "\n".join(
            ["{}\n  {}".format(key, blue(value.short_doc)) for
             key, value
             in
             runners.pipeline_dict.items()]))
