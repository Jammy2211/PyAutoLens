import numpy as np

from autolens.data import ccd
from autolens.data import convolution
from autolens.data.array import grids
from autolens.data.array import mask as msk
from test.unit.mock.mock_grids import MockBorders
from autolens.model.inversion import convolution as inversion_convolution

from test.unit.mock.mock_ccd import MockImage, MockNoiseMap, MockPSF
from test.unit.mock.mock_mask import MockMask, MockBlurringMask

class MockLensData(object):

    def __init__(self):

        self.pixel_scale = 2.0

        self.mask_2d = MockMask()
        self.image_2d = MockImage()
        self.noise_map_2d = MockNoiseMap()
        self.psf = MockPSF()

        self.sub_grid_size = 2

        self.grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=self.mask_2d, sub_grid_size=self.sub_grid_size, psf_shape=self.psf.shape)

        self.padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=self.mask_2d, sub_grid_size=self.sub_grid_size, psf_shape=self.psf.shape)

        self.border = MockBorders()

        self.image_1d = self.mask_2d.map_2d_array_to_masked_1d_array(array_2d=self.image_2d)
        self.noise_map_1d = self.mask_2d.map_2d_array_to_masked_1d_array(array_2d=self.noise_map_2d)
        self.mask_1d = self.mask_2d.map_2d_array_to_masked_1d_array(array_2d=self.mask_2d)

        self.blurring_mask = MockBlurringMask()

        self.convolver_image = convolution.ConvolverImage(mask=self.mask_2d, blurring_mask=self.blurring_mask,
                                                          psf=self.psf)

        self.convolver_mapping_matrix = inversion_convolution.ConvolverMappingMatrix(mask=self.mask_2d, psf=self.psf)

    @property
    def map_to_scaled_array(self):
        return self.grid_stack.scaled_array_2d_from_array_1d
