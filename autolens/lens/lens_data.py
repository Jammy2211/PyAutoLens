from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.data.convolution import ConvolverImage
from autolens.model.inversion import convolution as inversion_convolution


class LensData(object):

    def __init__(
            self,
            ccd_data,
            mask,
            sub_grid_size=2,
            positions=None,
            image_psf_shape=None,
            inversion_psf_shape=None,
            interp_pixel_scale=None,
            cluster_pixel_scale=None,
            cluster_pixel_limit=None,
            uses_inversion=True,
            uses_cluster_inversion=True
    ):
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
        cluster_pixel_limit : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        cluster_pixel_scale : float or None
            If *True*, the hyper image used to generate the cluster'grids weight map will be binned up to this higher \
            pixel scale to speed up the KMeans clustering algorithm.
        """

        self.ccd_data = ccd_data

        self.unmasked_image = ccd_data.image
        self.unmasked_noise_map = ccd_data.noise_map
        self.pixel_scale = ccd_data.pixel_scale
        self.psf = ccd_data.psf
        self.mask_1d = mask.array_1d_from_array_2d(array_2d=mask)
        self.image_1d = mask.array_1d_from_array_2d(array_2d=ccd_data.image)
        self.noise_map_1d = mask.array_1d_from_array_2d(
            array_2d=ccd_data.noise_map)

        self.sub_grid_size = sub_grid_size

        if image_psf_shape is None:
            self.image_psf_shape = self.psf.shape
        else:
            self.image_psf_shape = image_psf_shape

        self.convolver_image = ConvolverImage(
            mask=mask, blurring_mask=mask.blurring_mask_for_psf_shape(
                psf_shape=self.image_psf_shape),
            psf=self.psf.resized_scaled_array_from_array(
                new_shape=self.image_psf_shape))

        if inversion_psf_shape is None:
            self.inversion_psf_shape = self.psf.shape
        else:
            self.inversion_psf_shape = inversion_psf_shape

        self.uses_inversion = uses_inversion

        if uses_inversion:

            self.convolver_mapping_matrix = inversion_convolution.ConvolverMappingMatrix(
                mask=mask, psf=self.psf.resized_scaled_array_from_array(
                    new_shape=self.inversion_psf_shape))

        else:

            self.convolver_mapping_matrix = None

        self.grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=mask, sub_grid_size=sub_grid_size, psf_shape=self.image_psf_shape)

        self.padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=mask, sub_grid_size=sub_grid_size, psf_shape=self.image_psf_shape)

        self.interp_pixel_scale = interp_pixel_scale

        if interp_pixel_scale is not None:
            self.grid_stack = self.grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(
                interp_pixel_scale=interp_pixel_scale)

            self.padded_grid_stack = self.padded_grid_stack.new_grid_stack_with_interpolator_added_to_each_grid(
                interp_pixel_scale=interp_pixel_scale)

        self.border = grids.RegularGridBorder.from_mask(mask=mask)

        self.positions = positions

        self.mask_2d = mask
        self.image_2d = self.scaled_array_2d_from_array_1d(array_1d=self.image_1d)
        self.noise_map_2d = self.scaled_array_2d_from_array_1d(array_1d=self.noise_map_1d)

        self.uses_cluster_inversion = uses_cluster_inversion

        if uses_cluster_inversion:

            self.cluster_pixel_limit = cluster_pixel_limit
            self.cluster_pixel_scale = cluster_pixel_scale

            if self.cluster_pixel_scale is not None:

                self.cluster = grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
                    mask=self.mask_2d, cluster_pixel_scale=cluster_pixel_scale,
                    cluster_pixels_limit=cluster_pixel_limit)

            else:

                self.cluster = grids.ClusterGrid.from_mask_and_cluster_pixel_scale(
                    mask=self.mask_2d, cluster_pixel_scale=self.pixel_scale,
                    cluster_pixels_limit=cluster_pixel_limit)

        else:

            self.cluster = None
            self.cluster_pixel_limit = None
            self.cluster_pixel_scale = None

    def new_lens_data_with_modified_image(self, modified_image):

        ccd_data_with_modified_image = self.ccd_data.new_ccd_data_with_modified_image(
            modified_image=modified_image)

        return LensData(
            ccd_data=ccd_data_with_modified_image,
            mask=self.mask_2d,
            sub_grid_size=self.sub_grid_size,
            positions=self.positions,
            image_psf_shape=self.image_psf_shape,
            inversion_psf_shape=self.inversion_psf_shape,
            interp_pixel_scale=self.interp_pixel_scale,
            cluster_pixel_scale=self.cluster_pixel_scale,
            cluster_pixel_limit=self.cluster_pixel_limit,
            uses_inversion=self.uses_inversion,
            uses_cluster_inversion=self.uses_cluster_inversion)

    def new_lens_data_with_binned_up_ccd_data_and_mask(self, bin_up_factor):

        binned_up_ccd_data = self.ccd_data.new_ccd_data_with_binned_up_arrays(
            bin_up_factor=bin_up_factor)
        binned_up_mask = self.mask_2d.binned_up_mask_from_mask(
            bin_up_factor=bin_up_factor)

        return LensData(
            ccd_data=binned_up_ccd_data,
            mask=binned_up_mask,
            sub_grid_size=self.sub_grid_size,
            positions=self.positions,
            image_psf_shape=self.image_psf_shape,
            inversion_psf_shape=self.inversion_psf_shape,
            interp_pixel_scale=self.interp_pixel_scale,
            cluster_pixel_scale=self.cluster_pixel_scale,
            cluster_pixel_limit=self.cluster_pixel_limit,
            uses_inversion=self.uses_inversion,
            uses_cluster_inversion=self.uses_cluster_inversion)

    @property
    def array_1d_from_array_2d(self):
        return self.grid_stack.regular.array_1d_from_array_2d

    @property
    def scaled_array_2d_from_array_1d(self):
        return self.grid_stack.scaled_array_2d_from_array_1d

    def __array_finalize__(self, obj):
        if isinstance(obj, LensData):
            self.ccd_data = obj.ccd_data
            self.unmasked_image = obj.unmasked_image
            self.unmasked_noise_map = obj.unmasked_noise_map
            self.mask_2d = obj.mask_2d
            self.image_2d = obj.image_2d
            self.noise_map_2d = obj.noise_map_2d
            self.psf = obj.psf
            self.mask_1d = obj.mask_1d
            self.image_1d = obj.image_1d
            self.noise_map_1d = obj.noise_map_1d
            self.mask_1d = obj.mask_1d
            self.sub_grid_size = obj.sub_grid_size
            self.convolver_image = obj.convolver_image
            self.uses_inversion = obj.uses_inversion
            self.convolver_mapping_matrix = obj.convolver_mapping_matrix
            self.grid_stack = obj.grid_stack
            self.padded_grid_stack = obj.padded_grid_stack
            self.border = obj.border
            self.positions = obj.positions
            self.interp_pixel_scale = obj.interp_pixel_scale
            self.uses_cluster_inversion = obj.uses_cluster_inversion
            self.cluster = obj.cluster
            self.cluster_pixel_limit = obj.cluster_pixel_limit
