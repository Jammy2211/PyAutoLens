from autolens.array.mapping import reshape_returned_array
import numpy as np


class MockLensImagingData(object):
    def __init__(self, imaging_data, mask, grid, blurring_grid, convolver, binned_grid):

        self.imaging_data = imaging_data
        self.pixel_scale = imaging_data.pixel_scale

        self.psf = imaging_data.psf

        self.mask = mask
        self._mask_1d = self.mask.mapping.array_1d_from_array_2d(array_2d=self.mask)

        self.grid = grid
        self.grid.new_grid_with_binned_grid(binned_grid=binned_grid)
        self.sub_size = self.grid.sub_size
        self.convolver = convolver

        self._image_1d = self.mask.mapping.array_1d_from_array_2d(
            array_2d=imaging_data.image
        )
        self._noise_map_1d = self.mask.mapping.array_1d_from_array_2d(
            array_2d=imaging_data.noise_map
        )

        self.positions = None

        self.hyper_noise_map_max = None

        self.uses_cluster_inversion = False
        self.inversion_pixel_limit = 1000
        self.inversion_uses_border = True

        self.blurring_grid = blurring_grid
        self.preload_pixelization_grids_of_planes = None

    @property
    def mapping(self):
        return self.mask.mapping

    @reshape_returned_array
    def image(self, return_in_2d=True):
        return self._image_1d

    @reshape_returned_array
    def noise_map(self, return_in_2d=True):
        return self._noise_map_1d

    @reshape_returned_array
    def signal_to_noise_map(self, return_in_2d=True):
        return self._image_1d / self._noise_map_1d


class MockLensUVPlaneData(object):
    def __init__(self, uv_plane_data, mask, grid, transformer, binned_grid):

        self.uv_plane_data = uv_plane_data
        self.pixel_scale = uv_plane_data.pixel_scale

        self.mask = mask
        self._mask_1d = self.mask.mapping.array_1d_from_array_2d(array_2d=self.mask)

        self.grid = grid
        self.grid.new_grid_with_binned_grid(binned_grid=binned_grid)
        self.sub_size = self.grid.sub_size
        self.transformer = transformer

        self.positions = None

        self.hyper_noise_map_max = None

        self.uses_cluster_inversion = False
        self.inversion_pixel_limit = 1000
        self.inversion_uses_border = True

        self.preload_pixelization_grids_of_planes = None

    def visibilities(self):
        return self.uv_plane_data.visibilities

    @property
    def visibilities_mask(self):
        return np.full(fill_value=False, shape=self.uv_plane_data.uv_wavelengths.shape)

    def noise_map(self, return_x2=False):
        if not return_x2:
            return self.uv_plane_data.noise_map
        else:
            return np.stack(
                (self.uv_plane_data.noise_map, self.uv_plane_data.noise_map), axis=-1
            )

    @property
    def primary_beam(self):
        return self.uv_plane_data.primary_beam

    def signal_to_noise_map(self):
        return self.uv_plane_data.visibilities / self.uv_plane_data.noise_map
