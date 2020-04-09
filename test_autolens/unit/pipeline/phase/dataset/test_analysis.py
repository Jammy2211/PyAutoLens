from os import path

import pytest
from astropy import cosmology as cosmo
import numpy as np

import autofit as af
import autolens as al
from test_autolens.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestHyperMethods:
    def test__associate_images(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(redshift=0.5)
        galaxies.source = al.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        hyper_galaxy_image_path_dict = {
            ("galaxies", "lens"): np.ones((3, 3)),
            ("galaxies", "source"): 2.0 * np.ones((3, 3)),
        }

        results = mock_pipeline.MockResults(
            instance=instance,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            use_as_hyper_dataset=True,
        )

        analysis = al.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            cosmology=cosmo.Planck15,
            image_path="",
            results=results,
        )

        instance = analysis.associate_hyper_images(instance=instance)

        assert instance.galaxies.lens.hyper_galaxy_image == pytest.approx(
            np.ones((3, 3)), 1.0e-4
        )
        assert instance.galaxies.source.hyper_galaxy_image == pytest.approx(
            2.0 * np.ones((3, 3)), 1.0e-4
        )

        assert instance.galaxies.lens.hyper_model_image == pytest.approx(
            3.0 * np.ones((3, 3)), 1.0e-4
        )
        assert instance.galaxies.source.hyper_model_image == pytest.approx(
            3.0 * np.ones((3, 3)), 1.0e-4
        )
