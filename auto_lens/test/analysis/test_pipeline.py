from auto_lens.analysis import pipline


class MockImage:
    pass


class MockGalaxyPrior:
    def __init__(self):
        self.model_mapper = None

    def attach_to_model_mapper(self, model_mapper):
        self.model_mapper = model_mapper


class MockModelMapper:
    pass


class MockPixelization:
    pass


class MockNLO:
    pass


class TestAnalysis:
    def test_setup(self):
        lens_galaxy_prior = MockGalaxyPrior()
        source_galaxy_prior = MockGalaxyPrior()

        analysis = pipline.Analysis(image=MockImage(), lens_galaxy_priors=[lens_galaxy_prior],
                                    source_galaxy_priors=[source_galaxy_prior], model_mapper=MockModelMapper(),
                                    pixelization=MockPixelization(), non_linear_optimizer=MockNLO())

        assert lens_galaxy_prior.model_mapper is not None
        assert source_galaxy_prior.model_mapper is not None
