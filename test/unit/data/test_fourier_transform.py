from autolens.data import fourier_transform as ft

import numpy as np
import pytest

class TestRealVisiblities(object):

    def test__intensity_image_all_ones__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = np.ones(shape=(1,2))

        transformer = ft.Transformer(uv_wavelengths=uv_wavelengths, grid_radians=grid_radians)

        intensities_1d = np.ones(shape=(1))

        real_visibilities = transformer.real_visibilities_from_intensities(intensities_1d=intensities_1d)

        assert (real_visibilities == np.ones(shape=4)).all()

        uv_wavelengths = np.array([[0.2, 1.0],
                                   [0.5, 1.1],
                                   [0.8, 1.2]])

        grid_radians = np.array([[0.1, 0.2],
                                 [0.3, 0.4]])

        transformer = ft.Transformer(uv_wavelengths=uv_wavelengths, grid_radians=grid_radians)

        intensities_1d = np.ones(shape=(2))

        real_visibilities = transformer.real_visibilities_from_intensities(intensities_1d=intensities_1d)

        assert real_visibilities == pytest.approx(np.array([1.11715, 1.68257, 1.93716]), 1.0e-4)

    def test__intensity_image_varies__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = np.ones(shape=(1,2))

        transformer = ft.Transformer(uv_wavelengths=uv_wavelengths, grid_radians=grid_radians)

        intensities_1d = np.array([2.0])

        real_visibilities = transformer.real_visibilities_from_intensities(intensities_1d=intensities_1d)

        assert (real_visibilities == np.array([2.0])).all()

        uv_wavelengths = np.array([[0.2, 1.0],
                                   [0.5, 1.1],
                                   [0.8, 1.2]])

        grid_radians = np.array([[0.1, 0.2],
                                 [0.3, 0.4]])

        transformer = ft.Transformer(uv_wavelengths=uv_wavelengths, grid_radians=grid_radians)

        intensities_1d = np.array([3.0, 6.0])

        real_visibilities = transformer.real_visibilities_from_intensities(intensities_1d=intensities_1d)

        assert real_visibilities == pytest.approx(np.array([3.91361, 7.10136, 8.717248]), 1.0e-4)

    def test__preload_and_non_preload_give_same_answer(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = np.ones(shape=(1,2))

        transformer_preload = ft.Transformer(uv_wavelengths=uv_wavelengths, grid_radians=grid_radians, preload_transform=True)
        transformer = ft.Transformer(uv_wavelengths=uv_wavelengths, grid_radians=grid_radians, preload_transform=False)

        intensities_1d = np.array([2.0])

        real_visibilities_via_preload = transformer_preload.real_visibilities_from_intensities(intensities_1d=intensities_1d)
        real_visibilities = transformer.real_visibilities_from_intensities(intensities_1d=intensities_1d)

        assert (real_visibilities_via_preload == real_visibilities).all()