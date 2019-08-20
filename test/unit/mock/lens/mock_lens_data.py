from autolens.array.grids import reshape_array


class MockLensData(object):
    def __init__(self, ccd_data, mask, grid, blurring_grid, convolver, binned_grid):

        self.ccd_data = ccd_data
        self.unmasked_image = ccd_data.image
        self.unmasked_noise_map = ccd_data.noise_map
        self.pixel_scale = ccd_data.pixel_scale

        self.psf = ccd_data.psf

        self.mask_2d = mask
        self.mask_1d = self.mask_2d.array_1d_from_array_2d(array_2d=self.mask_2d)

        self.grid = grid
        self.grid.new_grid_with_binned_grid(binned_grid=binned_grid)
        self.sub_grid_size = self.grid.sub_grid_size
        self.convolver = convolver

        self.image_1d = self.mask_2d.array_1d_from_array_2d(
            array_2d=self.unmasked_image
        )
        self.noise_map_1d = self.mask_2d.array_1d_from_array_2d(
            array_2d=self.unmasked_noise_map
        )
        self.signal_to_noise_map_1d = self.image_1d / self.noise_map_1d

        self.positions = None

        self.hyper_noise_map_max = None

        self.uses_cluster_inversion = False
        self.inversion_pixel_limit = 1000
        self.inversion_uses_border = True

        self.preload_blurring_grid = blurring_grid
        self.preload_pixelization_grids_of_planes = None

    @reshape_array
    def image(self, return_in_2d=True):
        return self.image_1d

    @reshape_array
    def noise_map(self, return_in_2d=True):
        return self.noise_map_1d

    @reshape_array
    def signal_to_noise_map(self, return_in_2d=True):
        return self.signal_to_noise_map_1d

    @property
    def array_1d_from_array_2d(self):
        return self.grid.array_1d_from_array_2d

    @property
    def scaled_array_2d_from_array_1d(self):
        return self.grid.scaled_array_2d_from_array_1d
