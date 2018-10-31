from test.mock.mock_inversion import MockMapper
from test.mock.mock_galaxy import MockHyperGalaxy

class MockTracer(object):

    def __init__(self, images, blurring_images, has_light_profile, has_pixelization, has_hyper_galaxy,
                 has_grid_mappers=False):
        self.images = images
        self.blurring_images = blurring_images
        self.has_light_profile = has_light_profile
        self.has_pixelization = has_pixelization
        self.has_hyper_galaxy = has_hyper_galaxy
        self.has_grid_mappers = has_grid_mappers

    @property
    def all_planes(self):
        return []

    @property
    def image_plane_images_(self):
        return self.images

    @property
    def image_plane_blurring_images_(self):
        return self.blurring_images

    @property
    def mappers_of_planes(self):
        return [MockMapper()]

    @property
    def regularization_of_planes(self):
        return [MockMapper()]

    @property
    def hyper_galaxies(self):
        return [MockHyperGalaxy(), MockHyperGalaxy()]