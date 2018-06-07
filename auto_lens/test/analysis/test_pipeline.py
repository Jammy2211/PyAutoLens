from auto_lens.analysis import pipline


class MockImage:
    pass


class MockGalaxyPrior:
    pass


class MockPixelization:
    pass


class MockNLO:
    pass


class TestAnalysis:
    def test_setup(self):
        # image, lens_galaxy_priors, source_galaxy_priors, pixelization, non_linear_optimizer
        analysis = pipline.Analysis(image=MockImage(), lens_galaxy_priors=[MockGalaxyPrior()],
                                    source_galaxy_priors=[MockGalaxyPrior()], pixelization=MockPixelization(),
                                    non_linear_optimizer=MockNLO())
