from autolens.data import fourier_transform as ft

import numpy as np

class TestRealVisiblities(object):

    def test__simple_case(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = np.ones(shape=(1,2))

        transformer = ft.Transformer(uv_wavelengths=uv_wavelengths, grid_radians=grid_radians)

        intensities_1d = np.ones(shape=(1))

        real_visibilities = transformer.real_visibilities_from_intensities(intensities_1d=intensities_1d)

        assert (real_visibilities == np.ones(shape=4)).all()