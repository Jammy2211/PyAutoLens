from autolens.data.array import grids
from autolens.data import convolution
from autolens.data.array import mask as msk
from autolens.model.inversion import convolution as inversion_convolution


class LensData(object):

    def __init__(self, ccd_data, mask, sub_grid_size=2, image_psf_shape=None, inversion_psf_shape=None,
                 positions=None, interp_pixel_scale=None, optimal_sub_grid=True, uses_inversion=True):
        """
        The lens data is the collection of data (image, noise-map, PSF), a mask, grid_stack, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens data creates reduced 1D arrays of each \
        for lensing calculations.

        Parameters
        ----------
        ccd_data: im.CCD
            The ccd data all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_grid_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid_stack each image-pixel on a 2x2 \
            sub-grid.
        image_psf_shape : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        inversion_psf_shape : (int, int)
            The shape of the PSF used for convolving the inversion mapping matrix. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), \
            used to speed up the non-linear sampling.
        interp_pixel_scale : float
            If *True*, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the regular, sub and blurring grids.
        """

        self.ccd_data = ccd_data

        self.image = ccd_data.image
        self.noise_map = ccd_data.noise_map
        self.pixel_scale = ccd_data.pixel_scale
        self.psf = ccd_data.psf

        self.mask = mask

        self.image_1d = mask.map_2d_array_to_masked_1d_array(array_2d=ccd_data.image)
        self.noise_map_1d = mask.map_2d_array_to_masked_1d_array(array_2d=ccd_data.noise_map)
        self.mask_1d = mask.map_2d_array_to_masked_1d_array(array_2d=mask)

        self.sub_grid_size = sub_grid_size

        if image_psf_shape is None:
            self.image_psf_shape = self.psf.shape
        else:
            self.image_psf_shape = image_psf_shape

        self.convolver_image = convolution.ConvolverImage(mask=self.mask,
                                        blurring_mask=mask.blurring_mask_for_psf_shape(psf_shape=self.image_psf_shape),
                                        psf=self.psf.resized_scaled_array_from_array(new_shape=self.image_psf_shape))

        if inversion_psf_shape is None:
            self.inversion_psf_shape = self.psf.shape
        else:
            self.inversion_psf_shape = inversion_psf_shape

        self.uses_inversion = uses_inversion

        if uses_inversion:

            self.convolver_mapping_matrix = inversion_convolution.ConvolverMappingMatrix(
                 self.mask, self.psf.resized_scaled_array_from_array(new_shape=self.inversion_psf_shape))

        else:

            self.convolver_mapping_matrix = None

        self.optimal_sub_grid = optimal_sub_grid

        self.grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=mask, sub_grid_size=sub_grid_size, psf_shape=self.image_psf_shape,
            optimal_sub_grid=self.optimal_sub_grid)

        self.padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=mask, sub_grid_size=sub_grid_size, psf_shape=self.image_psf_shape,
            optimal_sub_grid=self.optimal_sub_grid)

        self.interp_pixel_scale = interp_pixel_scale

        if interp_pixel_scale is not None:

            self.grid_stack = self.grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(
                interp_pixel_scale=interp_pixel_scale)

            self.padded_grid_stack = self.padded_grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(
                interp_pixel_scale=interp_pixel_scale)

        self.border = grids.RegularGridBorder.from_mask(mask=mask)

        self.positions = positions

    def new_lens_data_with_modified_image(self, modified_image):

        ccd_data_with_modified_image = self.ccd_data.new_ccd_data_with_modified_image(modified_image=modified_image)

        return LensData(ccd_data=ccd_data_with_modified_image, mask=self.mask, sub_grid_size=self.sub_grid_size,
                        image_psf_shape=self.image_psf_shape, inversion_psf_shape=self.inversion_psf_shape,
                        positions=self.positions, interp_pixel_scale=self.interp_pixel_scale,
                        optimal_sub_grid=self.optimal_sub_grid, uses_inversion=self.uses_inversion)

    def new_lens_data_with_binned_up_ccd_data_and_mask(self, bin_up_factor):

        binned_up_ccd_data = self.ccd_data.new_ccd_data_with_binned_up_arrays(bin_up_factor=bin_up_factor)
        binned_up_mask = self.mask.binned_up_mask_from_mask(bin_up_factor=bin_up_factor)

        return LensData(ccd_data=binned_up_ccd_data, mask=binned_up_mask, sub_grid_size=self.sub_grid_size,
                        image_psf_shape=self.image_psf_shape, inversion_psf_shape=self.inversion_psf_shape,
                        positions=self.positions, interp_pixel_scale=self.interp_pixel_scale,
                        optimal_sub_grid=self.optimal_sub_grid, uses_inversion=self.uses_inversion)


    @property
    def map_to_scaled_array(self):
        return self.grid_stack.scaled_array_2d_from_array_1d

    def __array_finalize__(self, obj):
        if isinstance(obj, LensData):
            self.ccd_data = obj.ccd_data
            self.image = obj.image
            self.noise_map = obj.noise_map
            self.mask = obj.mask
            self.psf = obj.psf
            self.image_1d = obj.image_1d
            self.noise_map_1d = obj.noise_map_1d
            self.mask_1d = obj.mask_1d
            self.optimal_sub_grid = obj.optimal_sub_grid
            self.sub_grid_size = obj.sub_grid_size
            self.convolver_image = obj.convolver_image
            self.uses_inversion = obj.uses_inversion
            self.convolver_mapping_matrix = obj.convolver_mapping_matrix
            self.grid_stack = obj.grid_stack
            self.padded_grid_stack = obj.padded_grid_stack
            self.border = obj.border
            self.positions = obj.positions
            self.interp_pixel_scale = obj.interp_pixel_scale


class LensDataHyper(LensData):

    def __init__(self, ccd_data, mask, hyper_model_image, hyper_galaxy_images, hyper_minimum_values, sub_grid_size=2,
                 image_psf_shape=None, inversion_psf_shape=None, positions=None, interp_pixel_scale=None):
        """
        The lens data is the collection of data (image, noise-map, PSF), a mask, grid_stack, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens data creates reduced 1D arrays of each \
        for lensing calculations.

        Lens hyper-data includes the hyper-images necessary for changing different aspects of the data that is fitted \
        in a Bayesian framework, for example the background-sky subtraction and noise-map.

        Parameters
        ----------
        ccd_data: im.CCD
            The ccd data all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_grid_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid_stack each image-pixel on a 2x2 \
            sub-grid.
        image_psf_shape : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        inversion_psf_shape : (int, int)
            The shape of the PSF used for convolving the inversion mapping matrix. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), used \
            to speed up the non-linear sampling.
        interp_pixel_scale : float
            If *True*, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the regular, sub and blurring grids.
        """
        super().__init__(ccd_data=ccd_data, mask=mask, sub_grid_size=sub_grid_size, image_psf_shape=image_psf_shape,
                         inversion_psf_shape=inversion_psf_shape, positions=positions,
                         interp_pixel_scale=interp_pixel_scale)

        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_images = hyper_galaxy_images
        self.hyper_minimum_values = hyper_minimum_values

        self.hyper_model_image_1d = mask.map_2d_array_to_masked_1d_array(array_2d=hyper_model_image)
        self.hyper_galaxy_images_1d = list(map(lambda hyper_galaxy_image :
                                               mask.map_2d_array_to_masked_1d_array(hyper_galaxy_image),
                                               hyper_galaxy_images))

    def __array_finalize__(self, obj):
        super(LensDataHyper, self).__array_finalize__(obj)
        if isinstance(obj, LensDataHyper):
            self.hyper_model_image = obj.hyper_model_image
            self.hyper_galaxy_images = obj.hyper_galaxy_images
            self.hyper_minimum_values = obj.hyper_minimum_values
            self.hyper_model_image_1d = obj.hyper_model_image_1d
            self.hyper_galaxy_images_1d = obj.hyper_galaxy_images_1d