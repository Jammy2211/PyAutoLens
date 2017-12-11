from __future__ import division, print_function

import pytest
import light_profile
import profile


@pytest.fixture(name='circular')
def circular_sersic():
    return light_profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                            effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name='elliptical')
def elliptical_sersic():
    return light_profile.SersicLightProfile(axis_ratio=0.5, phi=0.0, flux=1.0,
                                            effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name='vertical')
def vertical_sersic():
    return light_profile.SersicLightProfile(axis_ratio=0.5, phi=90.0, flux=1.0,
                                            effective_radius=0.6, sersic_index=4.0)


@pytest.fixture(name='dev_vaucouleurs')
def dev_vaucouleurs_profile():
    return light_profile.DevVaucouleursLightProfile(centre=(0.0, 0.1), axis_ratio=0.6, phi=15.0, flux=2.0,
                                                    effective_radius=0.9)


@pytest.fixture(name="exponential")
def exponential_profile():
    return light_profile.ExponentialLightProfile(centre=(1, -1), axis_ratio=0.5, phi=45.0, flux=3.0,
                                                 effective_radius=0.2)


@pytest.fixture(name="core")
def core_profile():
    return light_profile.CoreSersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                                effective_radius=5, sersic_index=4.0, radius_break=0.01,
                                                flux_break=0.1, gamma=1, alpha=1)


class TestSetupProfiles(object):
    def test__setup_sersic__correct_values(self, circular):
        assert circular.x_cen == 0.0
        assert circular.y_cen == 0.0
        assert circular.axis_ratio == 1.0
        assert circular.phi == 0.0
        assert circular.flux == 1.0
        assert circular.effective_radius == 0.6
        assert circular.sersic_index == 4.0
        assert circular.sersic_constant == pytest.approx(7.66925, 1e-3)

    def test__setup_exponential__correct_values(self, exponential):
        assert exponential.x_cen == 1.0
        assert exponential.y_cen == -1.0
        assert exponential.axis_ratio == 0.5
        assert exponential.phi == 45.0
        assert exponential.flux == 3.0
        assert exponential.effective_radius == 0.2
        assert exponential.sersic_index == 1.0
        assert exponential.sersic_constant == pytest.approx(1.678378, 1e-3)

    def test__setup_dev_vaucouleurs__correct_values(self, dev_vaucouleurs):
        assert dev_vaucouleurs.x_cen == 0.0
        assert dev_vaucouleurs.y_cen == 0.1
        assert dev_vaucouleurs.axis_ratio == 0.6
        assert dev_vaucouleurs.phi == 15.0
        assert dev_vaucouleurs.flux == 2.0
        assert dev_vaucouleurs.effective_radius == 0.9
        assert dev_vaucouleurs.sersic_index == 4.0
        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66925, 1e-3)


class TestFluxValues(object):
    def test__flux_at_radius__correct_value(self, circular):
        flux_at_radius = circular.flux_at_radius(radius=1.0)
        assert flux_at_radius == pytest.approx(0.351797, 1e-3)

    def test__flux_at_radius_2__correct_value(self):
        sersic = light_profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=3.0,
                                                  effective_radius=2.0, sersic_index=2.0)
        flux_at_radius = sersic.flux_at_radius(
            radius=1.5)  # 3.0 * exp(-3.67206544592 * (1,5/2.0) ** (1.0 / 2.0)) - 1) = 0.351797
        assert flux_at_radius == pytest.approx(4.90657319276, 1e-3)

    def test__core_sersic_light_profile(self, core):
        assert core.flux_at_radius(0.01) == 0.1


class TestCoordinates(object):
    def test__coordinates_to_eccentric_radius(self, elliptical):
        assert elliptical.coordinates_to_eccentric_radius((1, 1)) == pytest.approx(
            elliptical.coordinates_to_eccentric_radius(
                (-1, -1)), 1e-10)

    def test__flux_at_coordinates(self, elliptical):
        assert elliptical.flux_at_coordinates((1, 1)) == pytest.approx(
            elliptical.flux_at_coordinates((-1, -1)), 1e-10)


class TestCombinedProfiles(object):
    def test__summation(self, circular):
        combined = light_profile.CombinedLightProfile(circular, circular)
        assert combined.flux_at_coordinates((0, 0)) == 2 * circular.flux_at_coordinates((0, 0))

    def test_1d_symmetry(self):
        sersic1 = light_profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                                   effective_radius=0.6, sersic_index=4.0)

        sersic2 = light_profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                                   effective_radius=0.6, sersic_index=4.0, centre=(100, 0))

        combined = light_profile.CombinedLightProfile(sersic1, sersic2)
        assert combined.flux_at_coordinates((0, 0)) == combined.flux_at_coordinates((100, 0))
        assert combined.flux_at_coordinates((49, 0)) == combined.flux_at_coordinates((51, 0))

    def test_2d_symmetry(self):
        sersic1 = light_profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                                   effective_radius=0.6, sersic_index=4.0)

        sersic2 = light_profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                                   effective_radius=0.6, sersic_index=4.0, centre=(100, 0))
        sersic3 = light_profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                                   effective_radius=0.6, sersic_index=4.0, centre=(0, 100))

        sersic4 = light_profile.SersicLightProfile(axis_ratio=1.0, phi=0.0, flux=1.0,
                                                   effective_radius=0.6, sersic_index=4.0, centre=(100, 100))

        combined = light_profile.CombinedLightProfile(sersic1, sersic2, sersic3, sersic4)

        assert combined.flux_at_coordinates((49, 0)) == pytest.approx(combined.flux_at_coordinates((51, 0)), 1e-5)
        assert combined.flux_at_coordinates((0, 49)) == pytest.approx(combined.flux_at_coordinates((0, 51)), 1e-5)
        assert combined.flux_at_coordinates((100, 49)) == pytest.approx(combined.flux_at_coordinates((100, 51)), 1e-5)
        assert combined.flux_at_coordinates((49, 49)) == pytest.approx(combined.flux_at_coordinates((51, 51)), 1e-5)


class TestEquivalentProfile(object):
    def test_as_sersic_profile(self, circular):
        copy = circular.as_sersic_profile()

        assert copy.centre == circular.centre
        assert copy.axis_ratio == circular.axis_ratio
        assert copy.phi == circular.phi
        assert copy.flux == circular.flux
        assert copy.sersic_index == circular.sersic_index

    def test_x_as_y(self, circular, exponential, dev_vaucouleurs, core):
        def assert_shared_base(x, y):
            assert x.centre == y.centre
            assert x.axis_ratio == y.axis_ratio
            assert x.phi == y.phi
            assert x.flux == y.flux

        assert_shared_base(circular, circular.as_exponential_profile())
        assert_shared_base(exponential, exponential.as_dev_vaucouleurs_profile())
        assert_shared_base(dev_vaucouleurs, dev_vaucouleurs.as_core_sersic_profile(1, 1, 1, 1))
        assert_shared_base(core, core.as_sersic_profile())


class TestArray(object):
    def test__simple_assumptions(self, circular):
        array = profile.array_function(circular.flux_at_coordinates)(x_min=0, x_max=101, y_min=0, y_max=101,
                                                                     pixel_scale=1)
        assert array.shape == (101, 101)
        assert array[51][51] > array[51][52]
        assert array[51][51] > array[52][51]
        assert all(map(lambda i: i > 0, array[0]))

        array = profile.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                     pixel_scale=0.5)
        assert array.shape == (200, 200)

    def test__ellipticity(self, circular, elliptical, vertical):
        array = profile.array_function(circular.flux_at_coordinates)(x_min=0, x_max=101, y_min=0, y_max=101,
                                                                     pixel_scale=1)
        assert array[60][0] == array[0][60]

        array = profile.array_function(elliptical.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                       pixel_scale=1)

        assert array[60][51] > array[51][60]

        array = profile.array_function(vertical.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                     pixel_scale=1)
        assert array[60][51] < array[51][60]

    # noinspection PyTypeChecker
    def test__flat_array(self, circular):
        array = profile.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                     pixel_scale=1)
        flat_array = profile.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                          pixel_scale=1).flatten()

        assert all(array[0] == flat_array[:100])
        assert all(array[1] == flat_array[100:200])

    def test_combined_array(self, circular):
        combined = light_profile.CombinedLightProfile(circular, circular)

        assert all(map(lambda i: i == 2,
                       profile.array_function(combined.flux_at_coordinates)().flatten() / profile.array_function(
                           circular.flux_at_coordinates)().flatten()))

    def test_symmetric_profile(self, circular):
        circular.centre = (50, 50)
        array = profile.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                     pixel_scale=1.0)

        assert array[50][50] > array[50][51]
        assert array[50][50] > array[49][50]
        assert array[49][50] == array[50][51]
        assert array[50][51] == array[50][49]
        assert array[50][49] == array[51][50]

        array = profile.array_function(circular.flux_at_coordinates)(x_min=0, x_max=100, y_min=0, y_max=100,
                                                                     pixel_scale=0.5)

        assert array[100][100] > array[100][101]
        assert array[100][100] > array[99][100]
        assert array[99][100] == array[100][101]
        assert array[100][101] == array[100][99]
        assert array[100][99] == array[101][100]

    def test_origin_symmetric_profile(self, circular):
        array = profile.array_function(circular.flux_at_coordinates)()

        assert circular.flux_at_coordinates((-5, 0)) < circular.flux_at_coordinates((0, 0))
        assert circular.flux_at_coordinates((5, 0)) < circular.flux_at_coordinates((0, 0))
        assert circular.flux_at_coordinates((0, -5)) < circular.flux_at_coordinates((0, 0))
        assert circular.flux_at_coordinates((0, 5)) < circular.flux_at_coordinates((0, 0))
        assert circular.flux_at_coordinates((5, 5)) < circular.flux_at_coordinates((0, 0))
        assert circular.flux_at_coordinates((-5, -5)) < circular.flux_at_coordinates((0, 0))

        assert array.shape == (100, 100)

        assert array[50][50] > array[50][51]
        assert array[50][50] > array[49][50]
        assert array[49][50] == pytest.approx(array[50][51], 1e-10)
        assert array[50][51] == pytest.approx(array[50][49], 1e-10)
        assert array[50][49] == pytest.approx(array[51][50], 1e-10)


class TestTransform(object):
    def test_exceptions(self, elliptical):
        with pytest.raises(profile.CoordinatesException):
            elliptical.coordinates_rotate_to_elliptical(profile.TransformedCoordinates((0, 0)))

        with pytest.raises(profile.CoordinatesException):
            elliptical.coordinates_back_to_cartesian((0, 0))
