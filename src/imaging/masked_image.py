from src.imaging import image as im
from src.pixelization import frame_convolution
import numpy as np


class MaskedImage(im.AbstractImage):
    def __new__(cls, image, mask):
        return np.array(mask.masked_1d_array_from_2d_array(image), ).view(cls)

    def __init__(self, image, mask, sub_grid_size=1):
        super().__init__(array=image,
                         effective_exposure_time=mask.masked_1d_array_from_2d_array(image.effective_exposure_time),
                         pixel_scale=image.pixel_scale,
                         psf=image.psf,
                         background_noise=mask.masked_1d_array_from_2d_array(image.background_noise),
                         poisson_noise=mask.masked_1d_array_from_2d_array(image.poisson_noise))
        self.border_pixel_indices = mask.border_pixel_indices
        self.coordinate_grid = mask.coordinate_grid
        self.blurring_mask = mask.blurring_mask_for_kernel_shape(image.psf.shape)
        self.blurring_coordinate_grid = self.blurring_mask.coordinate_grid
        self.kernel_convolver = frame_convolution.FrameMaker(mask).convolver_for_kernel(image.psf)
        self.sub_coordinate_grid = mask.sub_coordinate_grid_with_size(sub_grid_size)
        self.sub_to_image = mask.sub_to_image_with_size(sub_grid_size)
        self.sub_grid_size = sub_grid_size
        self.sub_grid_length = int(sub_grid_size ** 2.0)
        self.sub_grid_fraction = 1.0 / self.sub_grid_length
        self.grid_to_pixel = mask.grid_to_pixel()
        self.image_shape = image.shape

    def sub_data_to_image(self, data):
        return np.multiply(self.sub_grid_fraction, data.reshape(-1, self.sub_grid_length).sum(axis=1))

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
