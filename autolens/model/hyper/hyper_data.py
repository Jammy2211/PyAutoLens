class HyperImageSky(object):

    def __init__(self, background_sky_scale=0.0):
        """Class for scaling the background sky map and background noise_map-map of an regular.

        Parameters
        -----------
        background_sky_scale : float
            The value by which the background scale is increased or decreased (electrons per second).
        background_noise_scale : float
            The factor by which the background noise_maps is increased.
        """
        self.background_sky_scale = background_sky_scale

    def image_scaled_sky_from_image(self, image):
        """Compute a new regular with the background sky level scaled. This can simply multiple by a constant factor \
        (assuming a uniform background sky) because the regular is in units electrons per second.

        Parameters
        -----------
        image : ndarray
            The regular before scaling (electrons per second).
        """
        return image + self.background_sky_scale

class HyperNoiseBackground(object):

    def __init__(self, background_noise_scale=0.0):
        """Class for scaling the background sky map and background noise_map-map of an regular.

        Parameters
        -----------
        background_sky_scale : float
            The value by which the background scale is increased or decreased (electrons per second).
        background_noise_scale : float
            The factor by which the background noise_maps is increased.
        """
        self.background_noise_scale = background_noise_scale

    def noise_map_scaled_noise_from_noise_map(self, noise_map):
        """Compute a scaled noise_maps regular from the background noise_maps regular.

        Parameters
        -----------
        noise_map : ndarray
            The noise_maps before scaling (electrons per second).
        background_noise : ndarray
            The background noise_maps values (electrons per second).
        """
        return noise_map + self.background_noise_scale
