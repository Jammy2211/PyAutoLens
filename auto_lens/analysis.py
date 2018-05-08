
from auto_lens.imaging import grids
from auto_lens import ray_tracing

import numpy as np

class AnalysisData(object):

    def __init__(self, grid_datas, ray_tracing, grid_mappers):
        """ For a given ray-tracing model, compute the model images and use to fit the observed imaging data.

        Parameters
        ------------
        grid_datas : grids.GridDataCollection
            The observed imaging data to be fitted, converted to the *GridData* representation.
        ray_tracing : ray_tracing.TraceImageAndSource
            The ray-tracing configuration for the model galaxies and profiles.
        grid_mappers : grids.GridMapperCollection
            The mappers which map different grids to one another.
        """

        self.grid_datas = grid_datas
        self.ray_tracing = ray_tracing
        self.grid_mappers = grid_mappers

        self.galaxy_images = self.ray_tracing.generate_image_of_galaxies()
        self.galaxy_images_blurred = self.blurred_galaxy_images()

    def blurred_galaxy_images(self):
        """Blur the galaxy images"""
        self.grid_mappers.data_to_2d.map_to_2d(self.grid_datas.image.grid_data)
        self.grid_datas.psf.convolve_with_image(self.galaxy_images)