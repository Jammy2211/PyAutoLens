from autolens.data.array import grids
from test.unit.mock.data.mock_grids import MockBorders

from test.unit.mock.data import mock_ccd
from test.unit.mock.data.mock_mask import MockMask, MockBlurringMask

class MockLensData(object):

    def __init__(self):

        self.pixel_scale = 2.0

        self.mask_2d = MockMask()
        self.image_2d = mock_ccd.MockImage()
        self.noise_map_2d = mock_ccd.MockNoiseMap()
        self.psf = mock_ccd.MockPSF()

        self.sub_grid_size = 2

        self.grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=self.mask_2d, sub_grid_size=self.sub_grid_size, psf_shape=self.psf.shape)

        self.padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=self.mask_2d, sub_grid_size=self.sub_grid_size, psf_shape=self.psf.shape)

        self.border = MockBorders()
        self.blurring_mask = MockBlurringMask()
        self.convolver_image = mock_ccd.MockConvolverImage()
        self.convolver_mapping_matrix = mock_ccd.MockConvolverMappingMatrix()

        self.image_1d = mock_ccd.MockImage1D()
        self.noise_map_1d = mock_ccd.MockNoiseMap1D()
        self.mask_1d = self.mask_2d.map_2d_array_to_masked_1d_array(array_2d=self.mask_2d)

    @property
    def map_to_scaled_array(self):
        return self.grid_stack.scaled_array_2d_from_array_1d
