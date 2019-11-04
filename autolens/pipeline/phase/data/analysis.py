import autofit as af
from autolens.lens import ray_tracing


class Analysis(af.Analysis):
    def __init__(self, cosmology):

        self.cosmology = cosmology

    def hyper_image_sky_for_instance(self, instance):

        if hasattr(instance, "hyper_image_sky"):
            return instance.hyper_image_sky
        else:
            return None

    def hyper_background_noise_for_instance(self, instance):

        if hasattr(instance, "hyper_background_noise"):
            return instance.hyper_background_noise
        else:
            return None

    def tracer_for_instance(self, instance):
        return ray_tracing.Tracer.from_galaxies(
            galaxies=instance.galaxies, cosmology=self.cosmology
        )
