from src.imaging import image as im
from src.pixelization import frame_convolution
import numpy as np


class MaskedImage(im.AbstractImage):
    def __new__(cls, image, mask):
        return np.array(mask.map_to_1d(image), ).view(cls)

    def __init__(self, image, mask):
        super().__init__(array=image,
                         effective_exposure_time=mask.map_to_1d(image.effective_exposure_time),
                         pixel_scale=image.pixel_scale,
                         psf=image.psf,
                         background_noise=mask.map_to_1d(image.background_noise),
                         poisson_noise=mask.map_to_1d(image.poisson_noise))
        self.border_pixel_indices = mask.border_pixel_indices
        self.coordinate_grid = mask.coordinate_grid
        self.blurring_mask = mask.blurring_mask_for_kernel_shape(image.psf.shape)
        self.frame_maker = frame_convolution.FrameMaker(mask)
        self.convolver = self.frame_maker.convolver_for_kernel_shape(image.psf.shape, self.blurring_mask)
        self.kernel_convolver = self.convolver.convolver_for_kernel(image.psf)
        self.image_shape = image.shape
        self.image = image
        self.mask = mask
