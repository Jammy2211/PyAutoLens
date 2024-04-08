import copy

import autoarray as aa

from autolens.lens.tracer import Tracer
from autolens.imaging.fit_imaging import FitImaging
from autolens.analysis.result import ResultDataset


class ResultImaging(ResultDataset):

    @property
    def max_log_likelihood_fit(self) -> FitImaging:
        """
        An instance of a `FitImaging` corresponding to the maximum log likelihood model inferred by the non-linear
        search.
        """
        return self.analysis.fit_from(
            instance=self.instance,
        )

    @property
    def max_log_likelihood_tracer(self) -> Tracer:
        """
        An instance of a `Tracer` corresponding to the maximum log likelihood model inferred by the non-linear search.

        The `Tracer` is computed from the `max_log_likelihood_fit`, as this ensures that all linear light profiles
        are converted to normal light profiles with their `intensity` values updated.
        """
        return (
            self.max_log_likelihood_fit.model_obj_linear_light_profiles_to_light_profiles
        )

    @property
    def unmasked_model_image(self) -> aa.Array2D:
        """
        The model image of the maximum log likelihood model, created without using a mask.
        """
        return self.max_log_likelihood_fit.unmasked_blurred_image

    @property
    def unmasked_model_image_of_planes(self):
        """
        A list of the model image of every plane in the maximum log likelihood model, where all images are created
        without using a mask.
        """
        return self.max_log_likelihood_fit.unmasked_blurred_image_of_planes_list
