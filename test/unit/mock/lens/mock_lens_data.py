
class MockLensData(object):

    def __init__(self, ccd_data, mask, grid_stack, padded_grid_stack, border, convolver_image,
                 convolver_mapping_matrix):

        self.pixel_scale = ccd_data.pixel_scale

        self.image_2d = ccd_data.image
        self.noise_map_2d = ccd_data.noise_map
        self.psf = ccd_data.psf

        self.mask_2d = mask

        self.grid_stack = grid_stack
        self.padded_grid_stack = padded_grid_stack
        self.sub_grid_size = self.grid_stack.sub.sub_grid_size
        self.border = border
        self.convolver_image = convolver_image
        self.convolver_mapping_matrix = convolver_mapping_matrix

        self.image_1d = self.mask_2d.map_2d_array_to_masked_1d_array(array_2d=self.image_2d)
        self.noise_map_1d = self.mask_2d.map_2d_array_to_masked_1d_array(array_2d=self.noise_map_2d)
        self.mask_1d = self.mask_2d.map_2d_array_to_masked_1d_array(array_2d=self.mask_2d)

    @property
    def map_to_scaled_array(self):
        return self.grid_stack.scaled_array_2d_from_array_1d
