import numpy as np
import pytest

from autolens.data.imaging import image as im
from autolens.data.array import mask as msk
from autolens.data.fitting import fitting_data as fit_data, fitting
from autolens.model.galaxy import galaxy as g
from autolens.lensing import lensing_image
from test.mock.mock_galaxy import MockHyperGalaxy

@pytest.fixture(name='fi_no_blur')
def make_li_no_blur():

    image = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])

    psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0]])), pixel_scale=1.0, renormalize=False)
    image = im.Image(image, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = msk.Mask(array=ma, pixel_scale=1.0)

    return lensing_image.LensingImage(image, ma, sub_grid_size=2)


@pytest.fixture(name='fi_blur')
def make_li_blur():

    image = np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])

    psf = im.PSF(array=(np.array([[1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0],
                                     [1.0, 1.0, 1.0]])), pixel_scale=1.0, renormalize=False)
    image = im.Image(image, pixel_scale=1.0, psf=psf, noise_map=np.ones((4, 4)))

    ma = np.array([[True, True, True, True],
                   [True, False, False, True],
                   [True, False, False, True],
                   [True, True, True, True]])
    ma = msk.Mask(array=ma, pixel_scale=1.0)

    return lensing_image.LensingImage(image, ma, sub_grid_size=2)


