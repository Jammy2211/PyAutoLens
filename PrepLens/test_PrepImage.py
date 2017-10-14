from __future__ import division, print_function
import pytest
import numpy as np
from PrepImage import PrepImage

testdir = '/home/jammy/PycharmProjects/AutoLens/Data/testData/'

@pytest.fixture(scope='function')
def image3x3():
    image3x3 = PrepImage(workdir=testdir, file='3x3_ones.fits', hdu=0, pixel_scale=0.1)
    return image3x3

class TestClass:

    def test__init__input_image_3x3__all_attributes_correct(self):

        image = PrepImage(workdir=testdir, file='3x3_ones.fits', hdu=0, pixel_scale=0.1)

        assert (image.data2d == np.ones((3,3))).all()
        assert image.xy_dim[0] == 3
        assert image.xy_dim[1] == 3
        assert image.xy_arcsec[0] == pytest.approx(0.3)
        assert image.xy_arcsec[1] == pytest.approx(0.3)

    def test__set_mask_circular__input_big_mask__correct_mask(self, image3x3):

        image3x3.set_mask_circular(mask_radius_arcsec=0.5)

        assert (image3x3.mask2d == np.ones((3,3))).all()