import numpy as np

from autolens.data import ccd
from autolens.data import convolution
from autolens.data.array import grids
from autolens.data.array import mask as msk
from test.unit.mock.mock_grids import MockBorders
from autolens.model.inversion import convolution as inversion_convolution

class MockLensData(object):

    def __init__(self):

        self.pixel_scale = 2.0

        self.mask_2d = msk.Mask(np.array([[True, True, True, True, True, True],
                                          [True, True, True, True, True, True],
                                          [True, False, False, False, True, True],
                                          [True, False, False, False, True, True],
                                          [True, False, False, False, True, True],
                                          [True, True, True, True, True, True]]), pixel_scale=self.pixel_scale)

        self.image_2d = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
                                  [0.0, 4.0, 5.0, 6.0, 0.0, 0.0],
                                  [0.0, 7.0, 8.0, 9.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.noise_map_2d = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                      [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.psf = ccd.PSF(array=np.array([[1.0, 5.0, 9.0],
                                           [2.0, 5.0, 1.0],
                                           [3.0, 4.0, 0.0]]), pixel_scale=self.pixel_scale)

        self.sub_grid_size = 2

        self.grid_stack = grids.GridStack.grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=self.mask_2d, sub_grid_size=self.sub_grid_size, psf_shape=self.psf.shape)

        self.padded_grid_stack = grids.GridStack.padded_grid_stack_from_mask_sub_grid_size_and_psf_shape(
            mask=self.mask_2d, sub_grid_size=self.sub_grid_size, psf_shape=self.psf.shape)

        self.border = MockBorders()

        self.image_1d = self.mask_2d.map_2d_array_to_masked_1d_array(array_2d=self.image_2d)
        self.noise_map_1d = self.mask_2d.map_2d_array_to_masked_1d_array(array_2d=self.noise_map_2d)
        self.mask_1d = self.mask_2d.map_2d_array_to_masked_1d_array(array_2d=self.mask_2d)

        self.blurring_mask = msk.Mask(array=np.array([[True,  True,  True,  True,  True, True],
                                                     [False, False, False, False, False, True],
                                                     [False,  True,  True,  True, False, True],
                                                     [False,  True,  True,  True, False, True],
                                                     [False,  True,  True,  True, False, True],
                                                     [False, False, False, False, False,  True]]), pixel_scale=self.pixel_scale)

        self.convolver_image = convolution.ConvolverImage(mask=self.mask_2d, blurring_mask=self.blurring_mask,
                                                          psf=self.psf)

        self.convolver_mapping_matrix = inversion_convolution.ConvolverMappingMatrix(mask=self.mask_2d, psf=self.psf)

    @property
    def map_to_scaled_array(self):
        return self.grid_stack.scaled_array_2d_from_array_1d
