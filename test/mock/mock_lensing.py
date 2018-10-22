class MockTracer(object):

    def __init__(self, image, blurring_image, has_light_profile, has_pixelization, has_hyper_galaxy,
                 has_grid_mappers=False):
        self.image = image
        self.blurring_image = blurring_image
        self.has_light_profile = has_light_profile
        self.has_pixelization = has_pixelization
        self.has_hyper_galaxy = has_hyper_galaxy
        self.has_grid_mappers = has_grid_mappers

    @property
    def all_planes(self):
        return []

    @property
    def _image_plane_image(self):
        return self.image

    @property
    def _image_plane_blurring_image(self):
        return self.blurring_image

    @property
    def mappers_of_planes(self):
        return [MockMapper()]

    @property
    def regularization_of_planes(self):
        return [MockMapper()]

    @property
    def hyper_galaxies(self):
        return [MockHyperGalaxy(), MockHyperGalaxy()]