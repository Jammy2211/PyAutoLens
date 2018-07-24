from src.imaging import image as im
from src.imaging import mask as msk
from src.pixelization import frame_convolution
import numpy as np


class MaskedImage(im.AbstractImage):
    def __new__(cls, image, mask):
        return np.array(mask.masked_1d_array_from_2d_array(image), ).view(cls)

    def __init__(self, image, mask):
        """
        An image_coords that has been masked. Only data within the mask is kept. This data is kept in 1D with a corresponding
        array mapping data back to 2D.

        Parameters
        ----------
        image: im.Image
            A 2D image
        mask: msk.Mask
            A mask to be applied to the image_coords
        """
        super().__init__(array=image,
                         effective_exposure_time=mask.masked_1d_array_from_2d_array(image.effective_exposure_time),
                         pixel_scale=image.pixel_scale,
                         psf=image.psf,
                         background_noise=mask.masked_1d_array_from_2d_array(image.background_noise),
                         poisson_noise=mask.masked_1d_array_from_2d_array(image.poisson_noise))
        #  TODO: Kernel convolver and blurring_coords mask are here. Should they be?
        self.border_pixel_indices = mask.border_pixel_indices
        self.coordinate_grid = mask.coordinate_grid
        self.blurring_mask = mask.blurring_mask_for_kernel_shape(image.psf.shape)
        self.frame_maker = frame_convolution.FrameMaker(mask)
        self.convolver = self.frame_maker.convolver_for_kernel_shape(image.psf.shape, self.blurring_mask)
        self.kernel_convolver = self.convolver.convolver_for_kernel(image.psf)
        self.image_shape = image.shape
        self.image = image
        self.mask = mask

    def map_to_2d(self, data):
        """Use mapper to map an input data-set from a *GridData* to its original 2D image_coords.
        Parameters
        -----------
        data : ndarray
            The grid-data which is mapped to its 2D image_coords.
        """
        data_2d = np.zeros(self.image_shape)

        for (i, pixel) in enumerate(self.mask.grid_to_pixel()):
            data_2d[pixel[0], pixel[1]] = data[i]

        return data_2d