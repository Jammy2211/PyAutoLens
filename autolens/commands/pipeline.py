from autolens.commands.base import Base, current_directory
from autolens import conf


class Pipeline(Base):

    def run(self):
        from autolens import pipeline
        name = self.options['<name>']
        if name is not None:
            if name not in pipeline.pipeline_dict:
                print("No pipeline called '{}' found".format(name))
                return
            conf.instance = conf.Config(self.config_path, self.output_path)
            self.run_pipeline(pipeline.pipeline_dict[name]())

        print("Available Pipelines:\n")
        print("\n".join(list(pipeline.pipeline_dict.keys())))

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
