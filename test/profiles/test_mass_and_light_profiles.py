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
