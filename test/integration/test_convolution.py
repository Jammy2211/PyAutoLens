from autolens.imaging import mask
from autolens.imaging import image
from autolens.imaging import convolution

import numpy as np
import pytest


@pytest.fixture(name='sim_image_31x31', scope='function')
def sim_grid_9x9():
    sim_grid_9x9.ma = mask.Mask.for_simulate(shape_arc_seconds=(5.5, 5.5), pixel_scale=0.5, psf_size=(3, 3))
    sim_grid_9x9.image_grid = sim_grid_9x9.ma.coordinates_collection_for_subgrid_size_and_blurring_shape(
        sub_grid_size=1,
        blurring_shape=(3, 3))
    sim_grid_9x9.mapping = sim_grid_9x9.ma.grid_mapping_with_sub_grid_size(sub_grid_size=1, cluster_grid_size=1)
    return sim_grid_9x9


class TestConvolutuion:

    def test__compare_convolver_to_2d_convolution(self):
        # Setup a blurred image, using the PSF to perform the convolution in 2D, then mask it to make a 1d array.

        im = np.arange(900).reshape(30, 30)
        psf = image.PSF(array=np.arange(49).reshape(7, 7))
        blurred_im = psf.convolve(im)
        msk = mask.Mask.circular(shape_arc_seconds=(30.0, 30.0), pixel_scale=1.0, radius_mask=4.0)
        blurred_masked_im_0 = msk.map_to_1d(blurred_im)

        # Now reproduce this image using the frame convolver_image

        blurring_mask = msk.blurring_mask_for_kernel_shape(psf.shape)
        convolver = convolution.ConvolverImage(mask=msk, blurring_mask=blurring_mask, psf=psf)
        im_1d = msk.map_to_1d(im)
        blurring_im_1d = blurring_mask.map_to_1d(im)
        blurred_masked_im_1 = convolver.convolve_image_jit(image_array=im_1d, blurring_array=blurring_im_1d)

        assert blurred_masked_im_0 == pytest.approx(blurred_masked_im_1, 1e-4)
