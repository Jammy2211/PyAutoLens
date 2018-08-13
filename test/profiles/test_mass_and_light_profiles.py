from autolens.profiles import mass_and_light_profiles
import pytest
import numpy as np


class TestCase(object):
    class TestEllipticalSersic(object):
        def test__intensity_at_radius__correct_value(self):
            sersic = mass_and_light_profiles.EllipticalSersicMassAndLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                                                 effective_radius=0.6,
                                                                                 sersic_index=4.0)

            intensity = sersic.intensity_at_radius(radius=1.0)
            assert intensity == pytest.approx(0.351797, 1e-3)

        def test__flip_coordinates_lens_center__same_value(self):
            sersic = mass_and_light_profiles.EllipticalSersicMassAndLightProfile(centre=(0.0, 0.0), axis_ratio=1.0,
                                                                                 phi=0.0,
                                                                                 intensity=1.0,
                                                                                 effective_radius=1.0, sersic_index=4.0)

            defls_0 = sersic.deflections_from_grid(grid=np.array([[1.0, 1.0]]))

            sersic = mass_and_light_profiles.EllipticalSersicMassAndLightProfile(centre=(1.0, 1.0), axis_ratio=1.0,
                                                                                 phi=0.0,
                                                                                 intensity=1.0,
                                                                                 effective_radius=1.0, sersic_index=4.0)

            defls_1 = sersic.deflections_from_grid(grid=np.array([[0.0, 0.0]]))

            assert defls_0[0, 0] == pytest.approx(-defls_1[0, 0], 1e-5)
            assert defls_0[0, 1] == pytest.approx(-defls_1[0, 1], 1e-5)

    class TestEllipticalExponential(object):
        def test__intensity_at_radius__correct_value(self):
            exponential = mass_and_light_profiles.EllipticalExponentialMassAndLightProfile(axis_ratio=1.0, phi=0.0,
                                                                                           intensity=1.0,
                                                                                           effective_radius=0.6)

            intensity = exponential.intensity_at_radius(radius=1.0)
            assert intensity == pytest.approx(0.3266, 1e-3)

        def test__deflections_from_grid(self):
            exponential = mass_and_light_profiles.EllipticalExponentialMassAndLightProfile(centre=(-0.2, -0.4),
                                                                                           axis_ratio=0.8, phi=110.0,
                                                                                           intensity=5.0,
                                                                                           effective_radius=0.2,
                                                                                           mass_to_light_ratio=1.0)

            exponential_defls = exponential.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

            assert exponential_defls[0, 0] == pytest.approx(0.62569, 1e-3)
            assert exponential_defls[0, 1] == pytest.approx(0.90493, 1e-3)
