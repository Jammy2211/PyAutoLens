from test_autoarray.mock.mock_inversion import MockMapper
from test_autoastro.mock.mock_galaxy import MockHyperGalaxy


class MockTracer(object):
    def __init__(
        self,
        unblurred_image_1d,
        blurring_image_1d,
        has_light_profile,
        has_pixelization,
        has_hyper_galaxy,
        has_grid_mappers=False,
    ):

        self.unblurred_image_1d = unblurred_image_1d
        self.blurring_image_1d = blurring_image_1d
        self.has_light_profile = has_light_profile
        self.has_pixelization = has_pixelization
        self.has_hyper_galaxy = has_hyper_galaxy
        self.has_grid_mappers = has_grid_mappers

    @property
    def all_planes(self):
        return []

    @property
    def mappers_of_planes(self):
        return [MockMapper()]

    @property
    def regularization_of_planes(self):
        return [MockMapper()]

    @property
    def hyper_galaxies(self):
        return [MockHyperGalaxy(), MockHyperGalaxy()]
