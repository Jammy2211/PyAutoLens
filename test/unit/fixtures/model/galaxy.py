from autolens.model.galaxy import galaxy as g
from test.unit.fixtures.model.profiles import lp_0, lp_1, mp_0, mp_1

import pytest

@pytest.fixture()
def gal_x1_lp(lp_0):
    return g.Galaxy(redshift=0.5, lp0=lp_0)


@pytest.fixture()
def gal_x2_lp(lp_0, lp_1):
    return g.Galaxy(redshift=0.5, lp0=lp_0, lp1=lp_1)


@pytest.fixture()
def gal_x1_mp(mp_0):
    return g.Galaxy(redshift=0.5, mass_profile_0=mp_0)


@pytest.fixture()
def gal_x2_mp(mp_0, mp_1):
    return g.Galaxy(redshift=0.5, mass_profile_0=mp_0, mass_profile_1=mp_1)