from autolens.data.array.grids import reshape_returned_regular_array


class MockLensData(object):
    def __init__(
        self,
        ccd_data,
        mask,
        grid_stack,
        border,
        convolver_image,
        convolver_mapping_matrix,
        cluster,
    ):

        self.ccd_data = ccd_data
        self.unmasked_image = ccd_data.image
        self.unmasked_noise_map = ccd_data.noise_map
        self.pixel_scale = ccd_data.pixel_scale

        self.psf = ccd_data.psf

        self.mask_2d = mask
        self.mask_1d = self.mask_2d.array_1d_from_array_2d(array_2d=self.mask_2d)

        self.grid_stack = grid_stack
        self.sub_grid_size = self.grid_stack.sub.sub_grid_size
        self.border = border
        self.convolver_image = convolver_image
        self.convolver_mapping_matrix = convolver_mapping_matrix

        self.image_1d = self.mask_2d.array_1d_from_array_2d(
            array_2d=self.unmasked_image
        )
        self.noise_map_1d = self.mask_2d.array_1d_from_array_2d(
            array_2d=self.unmasked_noise_map
        )
        self.signal_to_noise_map_1d = self.image_1d / self.noise_map_1d

        self.positions = None

        self.cluster = cluster

        self.uses_cluster_inversion = False

    @reshape_returned_regular_array
    def image(self, return_in_2d=True):
        return self.image_1d

    @reshape_returned_regular_array
    def noise_map(self, return_in_2d=True):
        return self.noise_map_1d

    @reshape_returned_regular_array
    def signal_to_noise_map(self, return_in_2d=True):
        return self.signal_to_noise_map_1d

    @property
    def array_1d_from_array_2d(self):
        return self.grid_stack.regular.array_1d_from_array_2d

    @property
    def scaled_array_2d_from_array_1d(self):
        return self.grid_stack.scaled_array_2d_from_array_1d
