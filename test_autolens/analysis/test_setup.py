import autofit as af
import autolens as al

from autolens.mock import mock

import pytest


class TestSetupHyper:
    def test__hyper_galaxies_names_for_lens_and_source(self):

        setup = al.SetupHyper(hyper_galaxies_lens=False, hyper_galaxies_source=False)
        assert setup.hyper_galaxies is False
        assert setup.hyper_galaxy_names == None

        setup = al.SetupHyper(hyper_galaxies_lens=True, hyper_galaxies_source=False)
        assert setup.hyper_galaxies is True
        assert setup.hyper_galaxy_names == ["lens"]

        setup = al.SetupHyper(hyper_galaxies_lens=False, hyper_galaxies_source=True)
        assert setup.hyper_galaxies is True
        assert setup.hyper_galaxy_names == ["source"]

        setup = al.SetupHyper(hyper_galaxies_lens=True, hyper_galaxies_source=True)
        assert setup.hyper_galaxies is True
        assert setup.hyper_galaxy_names == ["lens", "source"]


class TestSetupMassLightDark:
    def test__update_stellar_mass_priors_using_einstein_radius(self):

        grid = al.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.05)

        tracer = mock.MockTracer(einstein_radius=1.0, einstein_mass=4.0)
        fit = mock.MockFit(grid=grid)

        result = mock.MockResult(
            max_log_likelihood_tracer=tracer, max_log_likelihood_fit=fit
        )

        bulge_prior_model = af.PriorModel(al.lmp.SphericalSersic)

        bulge_prior_model.intensity = af.UniformPrior(
            lower_limit=0.99, upper_limit=1.01
        )
        bulge_prior_model.effective_radius = af.UniformPrior(
            lower_limit=0.99, upper_limit=1.01
        )
        bulge_prior_model.sersic_index = af.UniformPrior(
            lower_limit=2.99, upper_limit=3.01
        )

        setup = al.SetupMassLightDark(bulge_prior_model=bulge_prior_model)

        setup.update_stellar_mass_priors_from_result(
            prior_model=bulge_prior_model,
            result=result,
            einstein_mass_range=[0.001, 10.0],
            bins=10,
        )

        assert setup.bulge_prior_model.mass_to_light_ratio.lower_limit == pytest.approx(
            0.00040519, 1.0e-1
        )
        assert setup.bulge_prior_model.mass_to_light_ratio.upper_limit == pytest.approx(
            4.051935, 1.0e-1
        )
