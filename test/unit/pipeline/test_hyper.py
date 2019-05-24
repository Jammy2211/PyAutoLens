from astropy import cosmology as cosmo

from autofit.mapper import model
from autolens.model import galaxy as g
from autolens.model.profiles import light_profiles as lp
from autolens.pipeline import phase as ph


class TestImagePassing(object):
    def test_galaxy_dict(self):
        lens_galaxies = model.ModelInstance()
        galaxy = g.Galaxy(light=lp.EllipticalLightProfile(), redshift=1.0)
        lens_galaxies.lens = galaxy

        instance = model.ModelInstance()
        instance.lens_galaxies = lens_galaxies

        result = ph.LensPlanePhase.Result(instance, 1.0, None, None,
                                          ph.LensPlanePhase.Analysis(lens_data=None, cosmology=cosmo.Planck15,
                                                                     positions_threshold=1.0), None)

        assert result.name_galaxy_tuples == [("lens", galaxy)]
