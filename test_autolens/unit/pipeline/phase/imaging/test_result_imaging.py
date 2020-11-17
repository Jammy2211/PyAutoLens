import autofit as af
import autolens as al
import numpy as np
from astropy import cosmology as cosmo
from autolens.mock import mock


class TestImagePassing:
    def test___image_dict(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(redshift=0.5)
        galaxies.source = al.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = al.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            settings=al.SettingsPhaseImaging(),
            results=mock.MockResults(),
            cosmology=cosmo.Planck15,
        )

        result = al.PhaseImaging.Result(
            samples=mock.MockSamples(max_log_likelihood_instance=instance),
            previous_model=af.ModelMapper(),
            analysis=analysis,
            search=None,
        )

        image_dict = result.image_galaxy_dict
        assert isinstance(image_dict[("galaxies", "lens")], np.ndarray)
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

        result.instance.galaxies.lens = al.Galaxy(redshift=0.5)

        image_dict = result.image_galaxy_dict
        assert (image_dict[("galaxies", "lens")].in_2d == np.zeros((7, 7))).all()
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

    def test__stochastic_log_evidences(self, masked_imaging_7x7):

        lens_hyper_image = al.Array.ones(shape_2d=(3, 3), pixel_scales=0.1)
        lens_hyper_image[4] = 10.0
        source_hyper_image = al.Array.ones(shape_2d=(3, 3), pixel_scales=0.1)
        source_hyper_image[4] = 10.0
        hyper_model_image = al.Array.full(
            fill_value=0.5, shape_2d=(3, 3), pixel_scales=0.1
        )

        hyper_galaxy_image_path_dict = {
            ("galaxies", "lens"): lens_hyper_image,
            ("galaxies", "source"): source_hyper_image,
        }

        results = mock.MockResults(
            use_as_hyper_dataset=True,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        galaxies = af.ModelInstance()
        galaxies.lens = al.Galaxy(
            redshift=0.5, mass=al.mp.SphericalIsothermal(einstein_radius=1.0)
        )
        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiMagnification(shape=(3, 3)),
            regularization=al.reg.Constant(),
        )

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = al.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            settings=al.SettingsPhaseImaging(),
            results=results,
            cosmology=cosmo.Planck15,
        )

        result = al.PhaseImaging.Result(
            samples=mock.MockSamples(max_log_likelihood_instance=instance),
            previous_model=af.ModelMapper(),
            analysis=analysis,
            search=None,
        )

        assert result.stochastic_log_evidences(histogram_samples=2) == None

        galaxies.source = al.Galaxy(
            redshift=1.0,
            pixelization=al.pix.VoronoiBrightnessImage(pixels=9),
            regularization=al.reg.Constant(),
        )

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = al.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            settings=al.SettingsPhaseImaging(),
            results=results,
            cosmology=cosmo.Planck15,
        )

        result = al.PhaseImaging.Result(
            samples=mock.MockSamples(max_log_likelihood_instance=instance),
            previous_model=af.ModelMapper(),
            analysis=analysis,
            search=None,
        )

        assert (
            result.stochastic_log_evidences(histogram_samples=2)[0]
            != result.stochastic_log_evidences(histogram_samples=2)[1]
        )
