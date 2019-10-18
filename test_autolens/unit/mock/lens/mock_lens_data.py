import numpy as np

class MockLensImagingData(object):
    def __init__(self, imaging, mask, grid, blurring_grid, convolver, binned_grid):
        self.imaging = imaging
        self.pixel_scales = imaging.pixel_scales

        self.psf = imaging.psf

        self.mapping = mask.mapping
        self.mask = mask
        self._mask_1d = self.mask.mapping.array_from_array_2d(array_2d=self.mask)

        self.grid = grid
        self.grid.binned = binned_grid
        self.sub_size = self.grid.sub_size
        self.convolver = convolver

        self._image_1d = self.mask.mapping.array_from_array_2d(
            array_2d=imaging.image
        )
        self._noise_map_1d = self.mask.mapping.array_from_array_2d(
            array_2d=imaging.noise_map
        )

        self.positions = None

        self.hyper_noise_map_max = None

        self.uses_cluster_inversion = False
        self.inversion_pixel_limit = 1000
        self.inversion_uses_border = True

        self.blurring_grid = blurring_grid
        self.preload_pixelization_grids_of_planes = None

    def image(self):
        return self._image_1d

    def noise_map(self):
        return self._noise_map_1d

    def signal_to_noise_map(self):
        return self._image_1d / self._noise_map_1d

    def check_positions_trace_within_threshold_via_tracer(self, tracer):
        pass

    def check_inversion_pixels_are_below_limit_via_tracer(self, tracer):
        pass


class MockLensUVPlaneData(object):
    def __init__(self, interferometer, mask, grid, transformer, binned_grid):

        self.interferometer = interferometer
        self.pixel_scales = interferometer.pixel_scales

        self.mask = mask
        self._mask_1d = self.mask.mapping.array_from_array_2d(array_2d=self.mask)

        self.grid = grid
        self.grid.binned = binned_grid
        self.sub_size = self.grid.sub_size
        self.transformer = transformer

        self.positions = None

        self.hyper_noise_map_max = None

        self.uses_cluster_inversion = False
        self.inversion_pixel_limit = 1000
        self.inversion_uses_border = True

        self.preload_pixelization_grids_of_planes = None

    def visibilities(self):
        return self.interferometer.visibilities

    @property
    def visibilities_mask(self):
        return np.full(fill_value=False, shape=self.interferometer.uv_wavelengths.shape)

    def noise_map(self, return_x2=False):
        if not return_x2:
            return self.interferometer.noise_map
        else:
            return np.stack(
                (self.interferometer.noise_map, self.interferometer.noise_map), axis=-1
            )

    @property
    def primary_beam(self):
        return self.interferometer.primary_beam

    def signal_to_noise_map(self):
        return self.interferometer.visibilities / self.interferometer.noise_map
