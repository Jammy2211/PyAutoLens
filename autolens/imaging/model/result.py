from autolens.lens.model.result import ResultDataset
from autolens.lens.model.preloads import Preloads


class ResultImaging(ResultDataset):
    @property
    def max_log_likelihood_fit(self):

        hyper_image_sky = self.analysis.hyper_image_sky_for_instance(
            instance=self.instance
        )

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=self.instance
        )

        return self.analysis.fit_imaging_for_tracer(
            tracer=self.max_log_likelihood_tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            preload_overwrite=Preloads(use_w_tilde=False),
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
