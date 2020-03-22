import autofit as af
from autoastro.galaxy import galaxy as g
from autolens.lens import ray_tracing


def last_result_with_use_as_hyper_dataset(results):

    if results is not None:
        for index, result in enumerate(reversed(results)):
            if result.use_as_hyper_dataset:
                return result


class Analysis(af.Analysis):
    def __init__(self, cosmology, results):

        self.cosmology = cosmology

        result = last_result_with_use_as_hyper_dataset(results=results)

        if result is not None:

            self.hyper_galaxy_image_path_dict = result.hyper_galaxy_image_path_dict
            self.hyper_model_image = result.hyper_model_image

        else:

            self.hyper_galaxy_image_path_dict = None
            self.hyper_model_image = None

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

    def associate_hyper_images(self, instance: af.ModelInstance) -> af.ModelInstance:
        """
        Takes images from the last result, if there is one, and associates them with galaxies in this phase
        where full-path galaxy names match.

        If the galaxy collection has a different name then an association is not made.

        e.g.
        galaxies.lens will match with:
            galaxies.lens
        but not with:
            galaxies.lens
            galaxies.source

        Parameters
        ----------
        instance
            A model instance with 0 or more galaxies in its tree

        Returns
        -------
        instance
           The input instance with images associated with galaxies where possible.
        """
        if self.hyper_galaxy_image_path_dict is not None:
            for galaxy_path, galaxy in instance.path_instance_tuples_for_class(
                g.Galaxy
            ):
                if galaxy_path in self.hyper_galaxy_image_path_dict:
                    galaxy.hyper_model_image = self.hyper_model_image
                    galaxy.hyper_galaxy_image = self.hyper_galaxy_image_path_dict[
                        galaxy_path
                    ]

        return instance
