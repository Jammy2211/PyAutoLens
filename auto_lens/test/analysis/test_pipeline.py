from auto_lens.analysis import pipeline
from auto_lens.analysis import galaxy_prior
from auto_lens.test import mock


class TestModelAnalysis:
    def test_setup(self):
        lens_galaxy_prior = galaxy_prior.GalaxyPrior()
        source_galaxy_prior = galaxy_prior.GalaxyPrior()

        model_mapper = mock.MockModelMapper()

        pipeline.ModelAnalysis(image=mock.MockImage(), lens_galaxy_priors=[lens_galaxy_prior],
                               source_galaxy_priors=[source_galaxy_prior], pixelization=mock.MockPixelization(),
                               model_mapper=model_mapper, non_linear_optimizer=mock.MockNLO())

        assert len(model_mapper.classes) == 2

    def test_run(self):
        pass

