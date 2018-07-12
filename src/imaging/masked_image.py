from src.imaging import image as im


class MaskedImage(im.AbstractImage):
    def __init__(self, mask, image):
        super().__init__(image,
                         mask.masked_1d_array_from_2d_array(image.effective_exposure_time),
                         image.pixel_scale,
                         image.psf,
                         mask.masked_1d_array_from_2d_array(image.background_noise),
                         mask.masked_1d_array_from_2d_array(image.poisson_noise))
        self.border_pixel_indices = mask.border_pixel_indices
        self.coordinate_grid = mask.coordinate_grid
        self.blurring_mask = mask.blurring_mask_for_kernel_shape(image.psf.shape)
