from autolens.data.array import grids
from autolens.data import convolution
from autolens.data.array import mask as msk
from autolens.model.inversion import convolution as inversion_convolution


class LensImageStack(object):

    def __init__(self, ccds, masks, sub_grid_size=2, image_psf_shape=None, mapping_matrix_psf_shape=None,
                 positions=None):
        """
        The lens image is the collection of data (regular, noise_map-maps, PSF), a masks, grid_stacks, convolvers \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image data is initially loaded in 2D, for the lens image the masked-image (and noise_map-maps) \
        are reduced to 1D arrays for faster calculations.

        Parameters
        ----------
        image: im.CCD
            The original image data in 2D.
        mask: msk.Mask
            The 2D masks that is applied to the image.
        sub_grid_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid_stacks each image-pixel on a 2x2 \
            sub-grid.
        image_psf_shape : (int, int)
            The shape of the PSF used for convolving model regular generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        mapping_matrix_psf_shape : (int, int)
            The shape of the PSF used for convolving the inversion mapping matrix. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), \
            used to speed up the non-linear sampling.
        """


        self.images = list(map(lambda ccd : ccd.image, ccds))
        self.noise_maps = list(map(lambda ccd : ccd.noise_map, ccds))
        self.pixel_scales = list(map(lambda ccd : ccd.pixel_scale, ccds))
        self.psfs = list(map(lambda ccd : ccd.psf, ccds))

        self.masks = masks

        self.images_1d = list(map(lambda image, mask :
                                  mask.map_2d_array_to_masked_1d_array(array_2d=image),
                                  self.images, self.masks))

        self.noise_maps_1d = list(map(lambda noise_map, mask :
                                  mask.map_2d_array_to_masked_1d_array(array_2d=noise_map),
                                  self.noise_maps, self.masks))

        self.masks_1d = list(map(lambda mask : mask.map_2d_array_to_masked_1d_array(array_2d=mask), self.masks))

        self.sub_grid_size = sub_grid_size

        if image_psf_shape is None:
            image_psf_shape = self.psfs[0].shape

        self.convolvers_image = list(map(lambda psf, mask :
                                convolution.ConvolverImage(mask=mask,
                                                           blurring_mask=mask.blurring_mask_for_psf_shape(psf_shape=image_psf_shape),
                                                           psf=psf.resized_scaled_array_from_array(new_shape=image_psf_shape)),
                                         self.psfs, self.masks))

        if mapping_matrix_psf_shape is None:
            mapping_matrix_psf_shape = self.psfs[0].shape

        self.convolvers_mapping_matrix = list(map(lambda psf, mask :
                                inversion_convolution.ConvolverMappingMatrix(mask=mask,
                                psf=psf.resized_scaled_array_from_array(new_shape=mapping_matrix_psf_shape)),
                                                  self.psfs, self.masks))

        self.grid_stacks = list(map(lambda mask :
                                    grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                    sub_grid_size=sub_grid_size, psf_shape=image_psf_shape),
                                    self.masks))

        self.padded_grid_stacks = list(map(lambda mask :
                                      grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(mask=mask,
                                      sub_grid_size=sub_grid_size, psf_shape=image_psf_shape),
                                      self.masks))

        self.borders = list(map(lambda mask : grids.RegularGridBorder.from_mask(mask=mask), self.masks))

        self.positions = positions

    @property
    def map_to_scaled_arrays(self):
        return list(map(lambda grid_stack : grid_stack.regular.scaled_array_from_array_1d, self.grid_stacks))

    def __array_finalize__(self, obj):
        if isinstance(obj, LensImageStack):
            self.images = obj.images
            self.noise_maps = obj.noise_maps
            self.masks = obj.masks
            self.psfs = obj.psfs
            self.images_1d = obj.images_1d
            self.noise_maps_1d = obj.noise_maps_1d
            self.masks_1d = obj.masks_1d
            self.sub_grid_size = obj.sub_grid_size
            self.convolvers_image = obj.convolvers_image
            self.convolvers_mapping_matrix = obj.convolvers_mapping_matrix
            self.grid_stacks = obj.grid_stacks
            self.padded_grid_stacks = obj.padded_grid_stacks
            self.borders = obj.borders
            self.positions = obj.positions


class LensHyperImageStack(LensImageStack):

    def __init__(self, ccds, masks, hyper_model_images, hyper_galaxy_images_stack, hyper_minimum_values,
                 sub_grid_size=2, image_psf_shape=None, mapping_matrix_psf_shape=None, positions=None):
        """
        The lens image is the collection of data (regular, noise_map-maps, PSF), a masks, grid_stacks, convolvers and other \
        utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image data is initially loaded in 2D, for the lens image the masked-image (and noise_map-maps) \
        are reduced to 1D arrays for faster calculations.

        Parameters
        ----------
        image: im.CCD
            The original image data in 2D.
        mask: msk.Mask
            The 2D masks that is applied to the image.
        sub_grid_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid_stacks each image-pixel on a 2x2 \
            sub-grid.
        image_psf_shape : (int, int)
            The shape of the PSF used for convolving model regular generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        mapping_matrix_psf_shape : (int, int)
            The shape of the PSF used for convolving the inversion mapping matrix. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), used \
            to speed up the non-linear sampling.
        """
        super().__init__(ccds=ccds, masks=masks, sub_grid_size=sub_grid_size, image_psf_shape=image_psf_shape,
                         mapping_matrix_psf_shape=mapping_matrix_psf_shape, positions=positions)

        self.hyper_model_images = hyper_model_images
        self.hyper_galaxy_images_stack = hyper_galaxy_images_stack
        self.hyper_minimum_values = hyper_minimum_values

        self.hyper_model_images_1d = list(map(lambda mask, hyper_model_image :
                                              mask.map_2d_array_to_masked_1d_array(array_2d=hyper_model_image),
                                              self.masks, self.hyper_model_images))

        self.hyper_galaxy_images_1d_stack = []

        for image_index in range(len(self.hyper_galaxy_images_stack)):

            self.hyper_galaxy_images_1d_stack.append(
                list(map(lambda mask, hyper_galaxy_image_stack :
                               mask.map_2d_array_to_masked_1d_array(array_2d=hyper_galaxy_image_stack),
                               self.masks, self.hyper_galaxy_images_stack[image_index])))

    def __array_finalize__(self, obj):
        super(LensHyperImageStack, self).__array_finalize__(obj)
        if isinstance(obj, LensHyperImageStack):
            self.hyper_model_images = obj.hyper_model_images
            self.hyper_galaxy_images_stack = obj.hyper_galaxy_images_stack
            self.hyper_minimum_values = obj.hyper_minimum_values
            self.hyper_model_images_1d = obj.hyper_model_images_1d
            self.hyper_galaxy_images_1d_stack = obj.hyper_galaxy_images_1d_stack