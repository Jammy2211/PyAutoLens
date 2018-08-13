from autolens.profiles import mass_and_light_profiles
import pytest


class TestCase(object):
    class TestEllipticalSersic(object):
        def test__intensity_at_radius__correct_value(self):
            sersic = mass_and_light_profiles.EllipticalSersicMassAndLightProfile(axis_ratio=1.0, phi=0.0, intensity=1.0,
                                                                                 effective_radius=0.6,
                                                                                 sersic_index=4.0)

            intensity = sersic.intensity_at_radius(radius=1.0)
            assert intensity == pytest.approx(0.351797, 1e-3)
