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
        self.blurring_coordinate_grid = self.blurring_mask.coordinate_grid
        self.frame_maker = frame_convolution.FrameMaker(mask)
        self.convolver = self.frame_maker.convolver_for_kernel_shape(image.psf.shape)
        self.kernel_convolver = self.convolver.convolver_for_kernel(image.psf)
        self.grid_to_pixel = mask.grid_to_pixel()
        self.image_shape = image.shape

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


class SubCoordinateGrid(np.ndarray):
    def __new__(cls, mask, *args, sub_grid_size=1, **kwargs):
        return np.array(mask.sub_coordinate_grid_with_size(sub_grid_size)).view(cls)

    def __init__(self, mask, sub_grid_size=1):
        # noinspection PyArgumentList
        super(SubCoordinateGrid, self).__init__()
        self.sub_to_image = mask.sub_to_image_with_size(sub_grid_size)
        self.sub_grid_size = sub_grid_size
        self.sub_grid_length = int(sub_grid_size ** 2.0)
        self.sub_grid_fraction = 1.0 / self.sub_grid_length

    def sub_data_to_image(self, data):
        return np.multiply(self.sub_grid_fraction, data.reshape(-1, self.sub_grid_length).sum(axis=1))
