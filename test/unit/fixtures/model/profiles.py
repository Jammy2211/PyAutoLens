import pytest

from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.profiles import light_and_mass_profiles as lmp

@pytest.fixture
def lp_0():
    return lp.SphericalSersic(intensity=1.0, effective_radius=2.0, sersic_index=2.0)

@pytest.fixture()
def lp_1():
    return lp.SphericalSersic(intensity=2.0, effective_radius=2.0, sersic_index=2.0)

@pytest.fixture()
def mp_0():
    return mp.SphericalIsothermal(einstein_radius=1.0)

@pytest.fixture()
def mp_1():
    return mp.SphericalIsothermal(einstein_radius=2.0)

@pytest.fixture()
def lmp_0():
    return lmp.EllipticalSersicRadialGradient()