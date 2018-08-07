from autolens.commands.base import Base, current_directory


class Pipeline(Base):

    def run(self):
        from autolens import pipeline
        name = self.options['<name>']
        if name is not None:
            try:
                self.run_pipeline(pipeline.pipeline_dict[name]())
            except KeyError:
                print("No pipeline called '{}' found".format(name))
                return

        print("Available Pipelines:")
        print("\n".join(list(pipeline.pipeline_dict.keys())))

    def run_pipeline(self, pipeline):
        pipeline.run(self.load_image())

    @property
    def image_path(self):
        image_path = self.options['<image_path>']
        if image_path is None:
            print("Please specify the path to the image folder")
            return 
        if not image_path.startswith("/"):
            image_path = "{}/{}".format(current_directory, image_path)
        return image_path

    @property
    def pixel_scale(self):
        return self.options['<pixel-scale>']

    def load_image(self):
        from autolens.imaging import scaled_array
        from autolens.imaging import image as im

        data = scaled_array.ScaledArray.from_fits(file_path='{}/image'.format(self.image_path), hdu=0,
                                                  pixel_scale=self.pixel_scale)
        noise = scaled_array.ScaledArray.from_fits(file_path='{}/noise'.format(self.image_path), hdu=0,
                                                   pixel_scale=self.pixel_scale)
        psf = im.PSF.from_fits(file_path='{}/psf'.format(self.image_path), hdu=0)

        return im.Image(array=data, pixel_scale=self.pixel_scale, psf=psf, noise=noise)
