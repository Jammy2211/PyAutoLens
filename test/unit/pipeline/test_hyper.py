import pytest
from astropy import cosmology as cosmo

from autofit.mapper import model
from autolens.model import galaxy as g
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.pipeline import phase as ph


@pytest.fixture(name="lens_galaxy")
def make_lens_galaxy():
    return g.Galaxy(mass=mp.EllipticalMassProfile(), redshift=1.0)


@pytest.fixture(name="lens_galaxies")
def make_lens_galaxies(lens_galaxy):
    lens_galaxies = model.ModelInstance()
    lens_galaxies.lens = lens_galaxy
    return lens_galaxies


class TestImagePassing(object):
    def test_lens_galaxy_dict(self, lens_galaxy, lens_galaxies):
        instance = model.ModelInstance()
        instance.lens_galaxies = lens_galaxies

        result = ph.LensPlanePhase.Result(instance, 1.0, None, None,
                                          ph.LensPlanePhase.Analysis(lens_data=None, cosmology=cosmo.Planck15,
                                                                     positions_threshold=1.0), None)

        assert result.name_galaxy_tuples == [("lens", lens_galaxy)]

    def test_lens_source_galaxy_dict(self, lens_galaxy):
        source_galaxies = model.ModelInstance()
        lens_galaxies = model.ModelInstance()
        source_galaxy = g.Galaxy(light=lp.EllipticalLightProfile(), redshift=1.0)
        source_galaxies.source = source_galaxy
        lens_galaxies.lens = lens_galaxy

        instance = model.ModelInstance()
        instance.source_galaxies = source_galaxies
        instance.lens_galaxies = lens_galaxies

        result = ph.LensSourcePlanePhase.Result(instance, 1.0, None, None,
                                                ph.LensSourcePlanePhase.Analysis(lens_data=None,
                                                                                 cosmology=cosmo.Planck15,
                                                                                 positions_threshold=1.0), None)

        assert result.name_galaxy_tuples == [("lens", lens_galaxy), ("source", source_galaxy)]
