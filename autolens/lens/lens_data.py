import numpy as np

from autolens.array import grids
from autolens.array import mask as msk
from autolens.data.convolution import Convolver
from autolens.array.grids import reshape_array


class LensData(object):
    def __init__(
        self,
        ccd_data,
        mask,
        sub_grid_size=2,
        positions=None,
        positions_threshold=None,
        trimmed_psf_shape=None,
        pixel_scale_interpolation_grid=None,
        pixel_scale_binned_grid=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
        hyper_noise_map_max=None,
        preload_pixelization_grids_of_planes=None,
    ):
        """
        The lens data is the collection of instrument (image, noise-map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens data creates reduced 1D arrays of each \
        for lensing calculations.

        Parameters
        ----------
        ccd_data: im.CCD
            The ccd instrument all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_grid_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        trimmed_psf_shape : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        inversion_psf_shape : (int, int)
            The shape of the PSF used for convolving the inversion mapping_util matrix. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), \
            used to speed up the non-linear sampling.
        pixel_scale_interpolation_grid : float
            If *True*, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the grid, sub and blurring grids.
        inversion_pixel_limit : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        pixel_scale_binned_cluster_grid : float or None
            If *True*, the hyper_galaxies image used to generate the cluster'grids weight map will be binned up to this higher \
            pixel scale to speed up the KMeans clustering algorithm.
        """

        ### CCD DATA + MASK ####

        self.ccd_data = ccd_data

        self.unmasked_image = ccd_data.image
        self.unmasked_noise_map = ccd_data.noise_map
        self.pixel_scale = ccd_data.pixel_scale
        self.psf = ccd_data.psf
        self.mask_1d = mask.array_1d_from_array_2d(array_2d=mask)
        self.image_1d = mask.array_1d_from_array_2d(array_2d=ccd_data.image)
        self.noise_map_1d = mask.array_1d_from_array_2d(array_2d=ccd_data.noise_map)
        self.mask_2d = mask
        self.sub_grid_size = sub_grid_size

        ### PSF TRIMMING + CONVOLVER ###

        if trimmed_psf_shape is None:
            self.trimmed_psf_shape = self.psf.shape
        else:
            self.trimmed_psf_shape = trimmed_psf_shape

        self.convolver = Convolver(
            mask=mask,
            blurring_mask=mask.blurring_mask_from_psf_shape(
                psf_shape=self.trimmed_psf_shape
            ),
            psf=self.psf.resized_scaled_array_from_array(
                new_shape=self.trimmed_psf_shape
            ),
        )

        ### GRIDS ###

        self.grid = grids.Grid.from_mask_and_sub_grid_size(
            mask=mask, sub_grid_size=sub_grid_size
        )

        self.pixel_scale_binned_grid = pixel_scale_binned_grid

        if pixel_scale_binned_grid is not None:
            binned_grid = grids.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
                mask=mask, pixel_scale_binned_grid=pixel_scale_binned_grid
            )
            self.grid.new_grid_with_binned_grid(binned_grid=binned_grid)

        self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid

        if pixel_scale_interpolation_grid is not None:

            self.grid = self.grid.new_grid_with_interpolator(
                pixel_scale_interpolation_grid=pixel_scale_interpolation_grid
            )

        ### POSITIONS ###

        if positions is not None:
            self.positions = list(
                map(lambda position_set: np.asarray(position_set), positions)
            )
        else:
            self.positions = positions

        self.positions_threshold = positions_threshold

        self.hyper_noise_map_max = hyper_noise_map_max

        self.inversion_pixel_limit = inversion_pixel_limit
        self.inversion_uses_border = inversion_uses_border

        self.preload_blurring_grid = grids.Grid.blurring_grid_from_mask_and_psf_shape(
            mask=mask, psf_shape=self.trimmed_psf_shape
        )

        if pixel_scale_interpolation_grid is not None:

            self.preload_blurring_grid = self.preload_blurring_grid.new_grid_with_interpolator(
                pixel_scale_interpolation_grid=pixel_scale_interpolation_grid
            )

        self.preload_pixelization_grids_of_planes = preload_pixelization_grids_of_planes

    def new_lens_data_with_modified_image(self, modified_image):

        ccd_data_with_modified_image = self.ccd_data.new_ccd_data_with_modified_image(
            modified_image=modified_image
        )

        return LensData(
            ccd_data=ccd_data_with_modified_image,
            mask=self.mask_2d,
            sub_grid_size=self.sub_grid_size,
            positions=self.positions,
            positions_threshold=self.positions_threshold,
            trimmed_psf_shape=self.trimmed_psf_shape,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            pixel_scale_binned_grid=self.pixel_scale_binned_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            hyper_noise_map_max=self.hyper_noise_map_max,
            preload_pixelization_grids_of_planes=self.preload_pixelization_grids_of_planes,
        )

    def new_lens_data_with_binned_up_ccd_data_and_mask(self, bin_up_factor):

        binned_up_ccd_data = self.ccd_data.new_ccd_data_with_binned_up_arrays(
            bin_up_factor=bin_up_factor
        )
        binned_up_mask = self.mask_2d.binned_up_mask_from_mask(
            bin_up_factor=bin_up_factor
        )

        return LensData(
            ccd_data=binned_up_ccd_data,
            mask=binned_up_mask,
            sub_grid_size=self.sub_grid_size,
            positions=self.positions,
            positions_threshold=self.positions_threshold,
            trimmed_psf_shape=self.trimmed_psf_shape,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            pixel_scale_binned_grid=self.pixel_scale_binned_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            hyper_noise_map_max=self.hyper_noise_map_max,
            preload_pixelization_grids_of_planes=self.preload_pixelization_grids_of_planes,
        )

    def new_lens_data_with_signal_to_noise_limit(self, signal_to_noise_limit):

        ccd_data_with_signal_to_noise_limit = self.ccd_data.new_ccd_data_with_signal_to_noise_limit(
            signal_to_noise_limit=signal_to_noise_limit
        )

        return LensData(
            ccd_data=ccd_data_with_signal_to_noise_limit,
            mask=self.mask_2d,
            sub_grid_size=self.sub_grid_size,
            positions=self.positions,
            positions_threshold=self.positions_threshold,
            trimmed_psf_shape=self.trimmed_psf_shape,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            pixel_scale_binned_grid=self.pixel_scale_binned_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            hyper_noise_map_max=self.hyper_noise_map_max,
            preload_pixelization_grids_of_planes=self.preload_pixelization_grids_of_planes,
        )

    def mask(self, return_in_2d=True):
        if return_in_2d:
            return self.mask_2d
        else:
            return self.mask_1d

    @reshape_array
    def image(self, return_in_2d=True):
        return self.image_1d

    @reshape_array
    def noise_map(self, return_in_2d=True):
        return self.noise_map_1d

    @property
    def signal_to_noise_map_1d(self):
        return self.image_1d / self.noise_map_1d

    @reshape_array
    def signal_to_noise_map(self, return_in_2d=True):
        return self.signal_to_noise_map_1d

    def __array_finalize__(self, obj):
        if isinstance(obj, LensData):
            self.ccd_data = obj.ccd_data
            self.unmasked_image = obj.unmasked_image
            self.unmasked_noise_map = obj.unmasked_noise_map
            self.mask_2d = obj.mask_2d
            self.mask_1d = obj.mask_1d
            self.psf = obj.psf
            self.trimmed_psf_shape = obj.trimmed_psf_shape
            self.mask_1d = obj.mask_1d
            self.image_1d = obj.image_1d
            self.noise_map_1d = obj.noise_map_1d
            self.sub_grid_size = obj.sub_grid_size
            self.convolver = obj.convolver
            self.grid = obj.grid
            self.positions = obj.positions
            self.pixel_scale_interpolation_grid = obj.pixel_scale_interpolation_grid
            self.pixel_scale_binned_grid = obj.pixel_scale_binned_grid
            self.inversion_uses_border = obj.inversion_uses_border
            self.inversion_pixel_limit = obj.inversion_pixel_limit
            self.hyper_noise_map_max = obj.hyper_noise_map_max
            self.preload_blurring_grid = obj.preload_blurring_grid
            self.preload_pixelization_grids_of_planes = (
                obj.preload_pixelization_grids_of_planes
            )
