from autolens.commands.base import Base, current_directory
from autolens import conf
from autolens import pipeline
import colorama


def color(text, fore):
    return "{}{}{}".format(fore, text, colorama.Fore.RESET)


def blue(text):
    return color(text, colorama.Fore.BLUE)


def red(text):
    return color(text, colorama.Fore.RED)


class Pipeline(Base):

    def run(self):
        name = self.options['<name>']
        if self.options['--info']:
            tup = pipeline.pipeline_dict[name]
            pl = tup.make()
            print(tup.doc)
            print(red("Phases:\n"))
            print("\n".join(["{}\n   {}".format(phase.__class__.__name__, blue(phase.doc())) for phase in pl.phases]))
            return
        if name is not None:
            if name not in pipeline.pipeline_dict:
                print("No pipeline called '{}' found".format(name))
                return
            conf.instance = conf.Config(self.config_path, self.output_path)
            self.run_pipeline(pipeline.pipeline_dict[name].make())

        print_pipelines()

    def run_pipeline(self, pipeline):
        pipeline.run(self.load_image())

    @property
    def image_path(self):
        image_path = self.options['--image']
        if image_path is None:
            print("Please specify the path to the image folder")
            return
        if not image_path.startswith("/"):
            image_path = "{}/{}".format(current_directory, image_path)
        return image_path

    @property
    def pixel_scale(self):
        return float(self.options['--pixel-scale'])

    @property
    def config_path(self):
        config_path = self.options['--config']
        if not conf.is_config(config_path):
            print("No config found at {}. Try running 'autolens download_config'".format(config_path))
            exit(1)
        return config_path

    @property
    def output_path(self):
        output_path = self.options['--output']
        if not output_path.startswith("/"):
            output_path = "{}/{}".format(current_directory, output_path)
        return output_path

    def load_image(self):
        from autolens.imaging import scaled_array
        from autolens.imaging import image as im

        data = scaled_array.ScaledArray.from_fits(file_path='{}/image'.format(self.image_path), hdu=0,
                                                  pixel_scale=self.pixel_scale)
        noise = scaled_array.ScaledArray.from_fits(file_path='{}/noise'.format(self.image_path), hdu=0,
                                                   pixel_scale=self.pixel_scale)
        psf = im.PSF.from_fits(file_path='{}/psf'.format(self.image_path), hdu=0)

        return im.Image(array=data, pixel_scale=self.pixel_scale, psf=psf, noise=noise)


def print_pipelines():
    print("Available Pipelines:\n")
    print(
        "\n".join(
            ["{}\n  {}".format(key, blue(value.short_doc)) for
             key, value
             in
             pipeline.pipeline_dict.items()]))
