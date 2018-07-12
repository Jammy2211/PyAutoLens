from src.imaging import image as im
import numpy as np


class MaskedImage(im.AbstractImage):
    def __new__(cls, image, mask):
        return np.array(mask.masked_1d_array_from_2d_array(image),).view(cls)

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
