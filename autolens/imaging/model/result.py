from autolens.lens.model.result import ResultDataset
from autolens.lens.model.preloads import Preloads


class ResultImaging(ResultDataset):
    @property
    def max_log_likelihood_fit(self):

        return self.analysis.fit_imaging_for_instance(
            instance=self.instance,
            preload_overwrite=Preloads(use_w_tilde=False),
            check_positions=False,
        )

    @property
    def unmasked_model_image(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image

    @property
    def unmasked_model_image_of_planes(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image_of_planes

    @property
    def unmasked_model_image_of_planes_and_galaxies(self):
        fit = self.max_log_likelihood_fit
        return fit.unmasked_blurred_image_of_planes_and_galaxies
