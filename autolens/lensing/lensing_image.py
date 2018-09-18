from autolens.imaging import image as im
from autolens.imaging import mask as msk
from autolens.imaging import convolution
import numpy as np


class LensingImage(im.Image):

    def __new__(cls, image, mask, sub_grid_size=2, image_psf_shape=None, mapping_matrix_psf_shape=None, positions=None):
        return np.array(mask.map_2d_array_to_masked_1d_array(image)).view(cls)

    def __init__(self, image, mask, sub_grid_size=2, image_psf_shape=None, mapping_matrix_psf_shape=None,
                 positions=None):
        """
        The lensing image is the collection of data (images, noise-maps, PSF), a mask, grids, convolvers and other \
        utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image data is initially loaded in 2D, for the lensing image the masked-image (and noise-maps) \
        are reduced to 1D arrays for faster calculations.

        Parameters
        ----------
        image: im.Image
            The original image data in 2D.
        mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_grid_size : int
            The size of the sub-grid used for each lensing SubGrid. E.g. a value of 2 grids each image-pixel on a 2x2 \
            sub-grid.
        image_psf_shape : (int, int)
            The shape of the PSF used for convolving model images generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        mapping_matrix_psf_shape : (int, int)
            The shape of the PSF used for convolving the pixelization mapping matrix. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that map close to one another in the source-plane(s), used \
            to speed up the non-linear sampling.
        """
        super().__init__(array=image, pixel_scale=image.pixel_scale,
                         noise_map=mask.map_2d_array_to_masked_1d_array(image.noise_map), psf=image.psf,
                         background_noise_map=image.background_noise_map)

        self.image = image
        self.mask = mask
        self.sub_grid_size = sub_grid_size

        if image_psf_shape is None:
            image_psf_shape = self.image.psf.shape
        if mapping_matrix_psf_shape is None:
            mapping_matrix_psf_shape = self.image.psf.shape

        print(image_psf_shape)
        self.blurring_mask = mask.blurring_mask_for_psf_shape(image_psf_shape)
        self.convolver_image = convolution.ConvolverImage(self.mask,
                                                          self.blurring_mask, self.image.psf.trim(image_psf_shape))
        self.convolver_mapping_matrix = convolution.ConvolverMappingMatrix(self.mask,
                                                                    self.image.psf.trim(mapping_matrix_psf_shape))
        self.grids = msk.ImagingGrids.grids_from_mask_sub_grid_size_and_blurring_shape(mask=mask,
                                                sub_grid_size=sub_grid_size, psf_shape=image_psf_shape)

        self.unmasked_grids = msk.ImagingGrids.unmasked_grids_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                                       sub_grid_size=sub_grid_size, psf_shape=image_psf_shape)

        self.borders = msk.ImagingGridBorders.from_mask_and_sub_grid_size(mask=mask, sub_grid_size=sub_grid_size)
        self.positions = positions

    def __array_finalize__(self, obj):
        super(LensingImage, self).__array_finalize__(obj)
        if isinstance(obj, LensingImage):
            self.image = obj.image
            self.mask = obj.mask
            self.blurring_mask = obj.blurring_mask
            self.convolver_image = obj.convolver_image
            self.convolver_mapping_matrix = obj.convolver_mapping_matrix
            self.grids = obj.grids
            self.borders = obj.borders
            self.positions = obj.positions