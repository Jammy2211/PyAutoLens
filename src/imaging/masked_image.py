from src.imaging import image as im
from src.pixelization import frame_convolution
import numpy as np


class MaskedImage(im.AbstractImage):
    def __new__(cls, image, mask):
        return np.array(mask.masked_1d_array_from_2d_array(image), ).view(cls)

    def __init__(self, image, mask):
        super().__init__(array=image,
                         effective_exposure_time=mask.masked_1d_array_from_2d_array(image.effective_exposure_time),
                         pixel_scale=image.pixel_scale,
                         psf=image.psf,
                         background_noise=mask.masked_1d_array_from_2d_array(image.background_noise),
                         poisson_noise=mask.masked_1d_array_from_2d_array(image.poisson_noise))
        self.border_pixel_indices = mask.border_pixel_indices
        self.coordinate_grid = mask.coordinate_grid
        self.blurring_mask = mask.blurring_mask_for_kernel_shape(image.psf.shape)
        self.frame_maker = frame_convolution.FrameMaker(mask)
        self.convolver = self.frame_maker.convolver_for_kernel_shape(image.psf.shape, self.blurring_mask)
        self.kernel_convolver = self.convolver.convolver_for_kernel(image.psf)
        self.grid_to_pixel = mask.grid_to_pixel()
        self.image_shape = image.shape
        self.image = image
        self.mask = mask

    def map_to_2d(self, grid_data):
        """Use mapper to map an input data-set from a *GridData* to its original 2D image.

        Parameters
        -----------
        grid_data : ndarray
            The grid-data which is mapped to its 2D image.
        """
        data_2d = np.zeros(self.image_shape)

        for (i, pixel) in enumerate(self.grid_to_pixel):
            data_2d[pixel[0], pixel[1]] = grid_data[i]

        return data_2d


# class CoordsCollection(object):
#
#     def __init__(self, image, sub, blurring):
#         """A collection of grids which contain the coordinates of an image. This includes the image's regular grid,
#         sub-gri, blurring region, etc.
#
#         Coordinate grids are passed through the ray-tracing module to set up the image, lens and source planes.
#
#         Parameters
#         -----------
#         image : GridCoordsImage
#             A grid of coordinates for the regular image grid.
#         sub : GridCoordsImageSub
#             A grid of coordinates for the sub-gridded image grid.
#         blurring : GridCoordsBlurring
#             A grid of coordinates for the blurring regions.
#         """
#
#         self.image = image
#         self.sub = sub
#         self.blurring = blurring
#
#     def deflection_grids_for_galaxies(self, galaxies):
#         """Compute the deflection angles of every grids (by integrating the mass profiles of the input galaxies)
#         and set these up as a new collection of grids."""
#
#         image = self.image.deflection_grid_for_galaxies(galaxies)
#         sub = self.sub.deflection_grid_for_galaxies(galaxies)
#         blurring = self.blurring.deflection_grid_for_galaxies(galaxies)
#
#         return CoordsCollection(image, sub, blurring)
#
#     def traced_grids_for_deflections(self, deflections):
#         """Setup a new collection of grids by tracing their coordinates using a set of deflection angles."""
#         image = self.image.ray_tracing_grid_for_deflections(deflections.image)
#         sub = self.sub.ray_tracing_grid_for_deflections(deflections.sub)
#         blurring = self.blurring.ray_tracing_grid_for_deflections(deflections.blurring)
#
#         return CoordsCollection(image, sub, blurring)
