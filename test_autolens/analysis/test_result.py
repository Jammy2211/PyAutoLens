import autofit as af
import autolens as al
import numpy as np
from astropy import cosmology as cosmo

from autolens.analysis import result as res
from autolens.mock import mock


class TestResultImaging:

    def test___image_dict(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(redshift=0.5)
        galaxies.source = al.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = al.AnalysisImaging(
            dataset=masked_imaging_7x7,
        )

        result = res.ResultImaging(
            samples=mock.MockSamples(max_log_likelihood_instance=instance),
            model=af.ModelMapper(),
            analysis=analysis,
            search=None,
        )

        image_dict = result.image_galaxy_dict
        assert isinstance(image_dict[("galaxies", "lens")], np.ndarray)
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

        result.instance.galaxies.lens = al.Galaxy(redshift=0.5)

        image_dict = result.image_galaxy_dict
        assert (image_dict[("galaxies", "lens")].native == np.zeros((7, 7))).all()
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)
