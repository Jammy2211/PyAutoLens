import cProfile
from profile import mass_profile
import pytest


def test_deflection_angles():
    sersic = mass_profile.SersicMassProfile(centre=(-0.2, -0.4), axis_ratio=0.8, phi=110.0, flux=5.0,
                                            effective_radius=0.2, sersic_index=2.0, mass_to_light_ratio=1.0)

    defls = sersic.deflection_angles_at_coordinates(coordinates=(0.1625, 0.1625))

    assert defls[0] == pytest.approx(0.79374, 1e-3)
    assert defls[1] == pytest.approx(1.1446, 1e-3)

    for i in range(20):
        for j in range(20):
            sersic.deflection_angles_at_coordinates(coordinates=(i * 0.01625, j * 0.01625))


cProfile.run('test_deflection_angles()')
