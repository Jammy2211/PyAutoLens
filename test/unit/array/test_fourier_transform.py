import autolens as al

import numpy as np
import pytest


class TestVisiblities(object):
    def test__real_visibilities__intensity_image_all_ones__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = np.ones(shape=(1, 2))

        transformer = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=False,
        )

        image_1d = np.ones(shape=(1))

        real_visibilities = transformer.real_visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert (real_visibilities == np.ones(shape=4)).all()

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = np.array([[0.1, 0.2], [0.3, 0.4]])

        transformer = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=False,
        )

        image_1d = np.ones(shape=(2))

        real_visibilities = transformer.real_visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert real_visibilities == pytest.approx(
            np.array([1.11715, 1.68257, 1.93716]), 1.0e-4
        )

    def test__real_visibilities__intensity_image_varies__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = np.ones(shape=(1, 2))

        transformer = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=False,
        )

        image_1d = np.array([2.0])

        real_visibilities = transformer.real_visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert (real_visibilities == np.array([2.0])).all()

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = np.array([[0.1, 0.2], [0.3, 0.4]])

        transformer = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=False,
        )

        image_1d = np.array([3.0, 6.0])

        real_visibilities = transformer.real_visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert real_visibilities == pytest.approx(
            np.array([3.91361, 7.10136, 8.717248]), 1.0e-4
        )

    def test__real_visibilities__preload_and_non_preload_give_same_answer(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = np.ones(shape=(1, 2))

        transformer_preload = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=True,
        )
        transformer = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=False,
        )

        image_1d = np.array([2.0])

        real_visibilities_via_preload = transformer_preload.real_visibilities_from_image_1d(
            image_1d=image_1d
        )
        real_visibilities = transformer.real_visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert (real_visibilities_via_preload == real_visibilities).all()

    def test__imaginary_visibilities__intensity_image_all_ones__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = np.ones(shape=(1, 2))

        transformer = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=False,
        )

        image_1d = np.ones(shape=(1))

        imaginary_visibilities = transformer.imaginary_visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert (imaginary_visibilities == np.zeros(shape=4)).all()

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = np.array([[0.1, 0.2], [0.3, 0.4]])

        transformer = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=False,
        )

        image_1d = np.ones(shape=(2))

        imaginary_visibilities = transformer.imaginary_visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert imaginary_visibilities == pytest.approx(
            np.array([1.350411, 0.791759, 0.0]), 1.0e-4
        )

    def test__imaginary_visibilities__intensity_image_varies__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = np.ones(shape=(1, 2))

        transformer = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=False,
        )

        image_1d = np.array([2.0])

        imaginary_visibilities = transformer.imaginary_visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert (imaginary_visibilities == np.array([0.0])).all()

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = np.array([[0.1, 0.2], [0.3, 0.4]])

        transformer = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=False,
        )

        image_1d = np.array([3.0, 6.0])

        imaginary_visibilities = transformer.imaginary_visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert imaginary_visibilities == pytest.approx(
            np.array([6.9980971, 4.56218, 0.746069]), 1.0e-4
        )

    def test__imaginary_visibilities__preload_and_non_preload_give_same_answer(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = np.ones(shape=(1, 2))

        transformer_preload = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=True,
        )
        transformer = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=False,
        )

        image_1d = np.array([2.0])

        imaginary_visibilities_via_preload = transformer_preload.imaginary_visibilities_from_image_1d(
            image_1d=image_1d
        )
        imaginary_visibilities = transformer.imaginary_visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert (imaginary_visibilities_via_preload == imaginary_visibilities).all()

    def test__visiblities_from_image__same_as_individual_calculations_above(self):

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = np.array([[0.1, 0.2], [0.3, 0.4]])

        transformer = al.Transformer(
            uv_wavelengths=uv_wavelengths,
            grid_radians=grid_radians,
            preload_transform=False,
        )

        image_1d = np.array([3.0, 6.0])

        visibilities = transformer.visibilities_from_image_1d(image_1d=image_1d)

        assert visibilities[:, 0] == pytest.approx(
            np.array([3.91361, 7.10136, 8.717248]), 1.0e-4
        )
        assert visibilities[:, 1] == pytest.approx(
            np.array([6.9980971, 4.56218, 0.746069]), 1.0e-4
        )

        real_visibilities = transformer.real_visibilities_from_image_1d(
            image_1d=image_1d
        )
        imaginary_visibilities = transformer.imaginary_visibilities_from_image_1d(
            image_1d=image_1d
        )

        assert (visibilities[:, 0] == real_visibilities).all()
        assert (visibilities[:, 1] == imaginary_visibilities).all()
