import numpy as np

from autolens.array import grids
from autolens.array import mask as msk
from autolens.array.convolution import Convolver
from autolens.array.fourier_transform import Transformer
from autolens.array.mapping import reshape_returned_array


class AbstractLensData(object):
    def __init__(
        self,
        mask,
        positions=None,
        positions_threshold=None,
        pixel_scale_interpolation_grid=None,
        pixel_scale_binned_grid=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
        hyper_noise_map_max=None,
        preload_pixelization_grids_of_planes=None,
    ):

        self.mask = mask
        self._mask_1d = mask.mapping.array_1d_from_array_2d(array_2d=mask)
        self.sub_size = mask.sub_size

        ### GRIDS ###

        self.grid = grids.Grid.from_mask(mask=mask)

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

        self.preload_pixelization_grids_of_planes = preload_pixelization_grids_of_planes

    @property
    def mapping(self):
        return self.mask.mapping

    @property
    def pixel_scale(self):
        return self.mask.pixel_scale


class LensImagingData(AbstractLensData):
    def __init__(
        self,
        imaging_data,
        mask,
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
        The lens data is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens data creates reduced 1D arrays of each \
        for lensing calculations.

        Parameters
        ----------
        imaging_data: im.Imaging
            The imaging data_type all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        trimmed_psf_shape : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
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

        self.imaging_data = imaging_data

        super(LensImagingData, self).__init__(
            mask=mask,
            positions=positions,
            positions_threshold=positions_threshold,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            pixel_scale_binned_grid=pixel_scale_binned_grid,
            inversion_pixel_limit=inversion_pixel_limit,
            inversion_uses_border=inversion_uses_border,
            hyper_noise_map_max=hyper_noise_map_max,
            preload_pixelization_grids_of_planes=preload_pixelization_grids_of_planes,
        )

        self._image_1d = self.image(return_in_2d=False)
        self._noise_map_1d = self.noise_map(return_in_2d=False)

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

        self.blurring_grid = grids.Grid.blurring_grid_from_mask_and_psf_shape(
            mask=mask, psf_shape=self.trimmed_psf_shape
        )

        if pixel_scale_interpolation_grid is not None:

            self.blurring_grid = self.blurring_grid.new_grid_with_interpolator(
                pixel_scale_interpolation_grid=pixel_scale_interpolation_grid
            )

    @reshape_returned_array
    def image(self, return_in_2d=True, return_masked=True):
        return self.imaging_data.image

    @reshape_returned_array
    def noise_map(self, return_in_2d=True, return_masked=True):
        return self.imaging_data.noise_map

    @property
    def psf(self):
        return self.imaging_data.psf

    @reshape_returned_array
    def signal_to_noise_map(self, return_in_2d=True, return_masked=True):
        return self.imaging_data.image / self.imaging_data.noise_map

    def new_lens_imaging_data_with_modified_image(self, modified_image):

        imaging_data_with_modified_image = self.imaging_data.new_imaging_data_with_modified_image(
            modified_image=modified_image
        )

        return LensImagingData(
            imaging_data=imaging_data_with_modified_image,
            mask=self.mask,
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

    def new_lens_imaging_data_with_binned_up_imaging_data_and_mask(self, bin_up_factor):

        binned_up_imaging_data = self.imaging_data.new_imaging_data_with_binned_up_arrays(
            bin_up_factor=bin_up_factor
        )
        binned_up_mask = self.mask.binned_up_mask_from_mask(bin_up_factor=bin_up_factor)

        return LensImagingData(
            imaging_data=binned_up_imaging_data,
            mask=binned_up_mask,
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

    def new_lens_imaging_data_with_signal_to_noise_limit(self, signal_to_noise_limit):

        imaging_data_with_signal_to_noise_limit = self.imaging_data.new_imaging_data_with_signal_to_noise_limit(
            signal_to_noise_limit=signal_to_noise_limit
        )

        return LensImagingData(
            imaging_data=imaging_data_with_signal_to_noise_limit,
            mask=self.mask,
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

    def __array_finalize__(self, obj):
        if isinstance(obj, LensImagingData):
            self.imaging_data = obj.imaging_data
            self.mask = obj.mask
            self._mask_1d = obj._mask_1d
            self._image_1d = obj._image_1d
            self._noise_map_1d = obj._noise_map_1d
            self.trimmed_psf_shape = obj.trimmed_psf_shape
            self.sub_size = obj.sub_size
            self.convolver = obj.convolver
            self.grid = obj.grid
            self.positions = obj.positions
            self.pixel_scale_interpolation_grid = obj.pixel_scale_interpolation_grid
            self.pixel_scale_binned_grid = obj.pixel_scale_binned_grid
            self.inversion_uses_border = obj.inversion_uses_border
            self.inversion_pixel_limit = obj.inversion_pixel_limit
            self.hyper_noise_map_max = obj.hyper_noise_map_max
            self.blurring_grid = obj.blurring_grid
            self.preload_pixelization_grids_of_planes = (
                obj.preload_pixelization_grids_of_planes
            )


class LensUVPlaneData(AbstractLensData):
    def __init__(
        self,
        uv_plane_data,
        mask,
        positions=None,
        positions_threshold=None,
        trimmed_primary_beam_shape=None,
        pixel_scale_interpolation_grid=None,
        pixel_scale_binned_grid=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
        hyper_noise_map_max=None,
        preload_pixelization_grids_of_planes=None,
    ):
        """
        The lens data is the collection of data_type (image, noise-map, primary_beam), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens data creates reduced 1D arrays of each \
        for lensing calculations.

        Parameters
        ----------
        imaging_data: im.Imaging
            The imaging data_type all in 2D (the image, noise-map, primary_beam, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        trimmed_primary_beam_shape : (int, int)
            The shape of the primary_beam used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the primary_beam relative to the input image primary_beam, giving a faster analysis run-time.
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

        self.uv_plane_data = uv_plane_data

        super(LensUVPlaneData, self).__init__(
            mask=mask,
            positions=positions,
            positions_threshold=positions_threshold,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            pixel_scale_binned_grid=pixel_scale_binned_grid,
            inversion_pixel_limit=inversion_pixel_limit,
            inversion_uses_border=inversion_uses_border,
            hyper_noise_map_max=hyper_noise_map_max,
            preload_pixelization_grids_of_planes=preload_pixelization_grids_of_planes,
        )

        if trimmed_primary_beam_shape is None and self.primary_beam is not None:
            self.trimmed_primary_beam_shape = self.primary_beam.shape
        elif self.primary_beam is None:
            self.trimmed_primary_beam_shape = None
        else:
            self.trimmed_primary_beam_shape = trimmed_primary_beam_shape

        self.transformer = Transformer(
            uv_wavelengths=uv_plane_data.uv_wavelengths,
            grid_radians=self.grid.in_radians,
        )

    def visibilities(self):
        return self.uv_plane_data.visibilities

    def noise_map(self, return_x2=False):
        if not return_x2:
            return self.uv_plane_data.noise_map
        else:
            return np.stack(
                (self.uv_plane_data.noise_map, self.uv_plane_data.noise_map), axis=-1
            )

    @property
    def visibilities_mask(self):
        return np.full(fill_value=False, shape=self.uv_plane_data.uv_wavelengths.shape)

    @property
    def primary_beam(self):
        return self.uv_plane_data.primary_beam

    def signal_to_noise_map(self):
        return self.uv_plane_data.visibilities / self.uv_plane_data.noise_map

    def new_lens_imaging_data_with_modified_visibilities(self, modified_visibilities):

        uv_plane_data_with_modified_visibilities = self.uv_plane_data.new_uv_plane_data_with_modified_visibilities(
            modified_visibilities=modified_visibilities
        )

        return LensUVPlaneData(
            uv_plane_data=uv_plane_data_with_modified_visibilities,
            mask=self.mask,
            positions=self.positions,
            positions_threshold=self.positions_threshold,
            trimmed_primary_beam_shape=self.trimmed_primary_beam_shape,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            pixel_scale_binned_grid=self.pixel_scale_binned_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            hyper_noise_map_max=self.hyper_noise_map_max,
            preload_pixelization_grids_of_planes=self.preload_pixelization_grids_of_planes,
        )

    def __array_finalize__(self, obj):
        if isinstance(obj, LensUVPlaneData):
            self.uv_plane_data = obj.uv_plane_data
            self.mask = obj.mask
            self.trimmed_primary_beam_shape = obj.trimmed_primary_beam_shape
            self.sub_size = obj.sub_size
            self.transformer = obj.transformer
            self.grid = obj.grid
            self.positions = obj.positions
            self.pixel_scale_interpolation_grid = obj.pixel_scale_interpolation_grid
            self.pixel_scale_binned_grid = obj.pixel_scale_binned_grid
            self.inversion_uses_border = obj.inversion_uses_border
            self.inversion_pixel_limit = obj.inversion_pixel_limit
            self.hyper_noise_map_max = obj.hyper_noise_map_max
            self.preload_pixelization_grids_of_planes = (
                obj.preload_pixelization_grids_of_planes
            )
