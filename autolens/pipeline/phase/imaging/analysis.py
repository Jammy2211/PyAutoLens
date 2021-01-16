from autoconf import conf
import autofit as af
from autoarray.inversion import pixelizations as pix
from autoarray.exc import PixelizationException, InversionException, GridException
from autofit.exc import FitException
from autogalaxy.pipeline.phase.dataset import analysis as ag_analysis
from autolens.fit import fit
from autolens.pipeline import visualizer as vis
from autolens.pipeline.phase.dataset import analysis as analysis_dataset
from autogalaxy.pipeline.phase.imaging.analysis import Attributes as AgAttributes

import numpy as np
import copy


class Analysis(ag_analysis.Analysis, analysis_dataset.Analysis):
    def __init__(self, masked_imaging, settings, cosmology, results=None):

        super().__init__(
            masked_dataset=masked_imaging,
            settings=settings,
            cosmology=cosmology,
            results=results,
        )

    @property
    def masked_imaging(self):
        return self.masked_dataset

    def log_likelihood_function(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the masked_imaging in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model masked_imaging itself
        """

        self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        self.settings.settings_lens.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer, positions=self.masked_dataset.positions
        )

        self.settings.settings_lens.check_einstein_radius_with_threshold_via_tracer(
            tracer=tracer, grid=self.masked_dataset.grid
        )

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        if self.settings.settings_lens.stochastic_likelihood_resamples is None:

            try:
                return self.masked_imaging_fit_for_tracer(
                    tracer=tracer,
                    hyper_image_sky=hyper_image_sky,
                    hyper_background_noise=hyper_background_noise,
                ).figure_of_merit
            except (
                PixelizationException,
                InversionException,
                GridException,
                OverflowError,
            ) as e:
                raise FitException from e

        else:

            figures_of_merit = []

            for i in range(self.settings.settings_lens.stochastic_likelihood_resamples):

                settings_pixelization = copy.deepcopy(
                    self.settings.settings_pixelization
                )

                settings_pixelization.kmeans_seed = i
                #       settings_pixelization.is_stochastic = True

                try:
                    figures_of_merit.append(
                        fit.FitImaging(
                            masked_imaging=self.masked_dataset,
                            tracer=tracer,
                            hyper_image_sky=hyper_image_sky,
                            hyper_background_noise=hyper_background_noise,
                            settings_pixelization=settings_pixelization,
                            settings_inversion=self.settings.settings_inversion,
                        ).log_evidence
                    )
                except (
                    PixelizationException,
                    InversionException,
                    GridException,
                    OverflowError,
                ) as e:
                    raise FitException from e

            return np.mean(figures_of_merit)

    def masked_imaging_fit_for_tracer(
        self, tracer, hyper_image_sky, hyper_background_noise, use_hyper_scalings=True
    ):

        return fit.FitImaging(
            masked_imaging=self.masked_dataset,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scaling=use_hyper_scalings,
            settings_pixelization=self.settings.settings_pixelization,
            settings_inversion=self.settings.settings_inversion,
        )

    def stochastic_log_evidences_for_instance(self, instance):

        instance = self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        if not tracer.has_pixelization:
            return

        if not isinstance(
            tracer.pixelizations_of_planes[-1], pix.VoronoiBrightnessImage
        ):
            return

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        settings_pixelization = (
            self.settings.settings_pixelization.settings_with_is_stochastic_true()
        )

        log_evidences = []

        for i in range(self.settings.settings_lens.stochastic_samples):

            try:
                log_evidence = fit.FitImaging(
                    masked_imaging=self.masked_dataset,
                    tracer=tracer,
                    hyper_image_sky=hyper_image_sky,
                    hyper_background_noise=hyper_background_noise,
                    settings_pixelization=settings_pixelization,
                    settings_inversion=self.settings.settings_inversion,
                ).log_evidence
            except (
                PixelizationException,
                InversionException,
                GridException,
                OverflowError,
            ) as e:
                log_evidence = None

            if log_evidence is not None:
                log_evidences.append(log_evidence)

        return log_evidences

    def visualize(self, paths: af.Paths, instance, during_analysis):

        instance = self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)
        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)
        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.masked_imaging_fit_for_tracer(
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        visualizer = vis.Visualizer(visualize_path=paths.image_path)

        visualizer.visualize_imaging(imaging=self.masked_imaging.imaging)
        visualizer.visualize_fit_imaging(fit=fit, during_analysis=during_analysis)
        visualizer.visualize_tracer(
            tracer=fit.tracer, grid=fit.grid, during_analysis=during_analysis
        )
        if fit.inversion is not None:
            visualizer.visualize_inversion(
                inversion=fit.inversion, during_analysis=during_analysis
            )

        visualizer.visualize_hyper_images(
            hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
            hyper_model_image=self.hyper_model_image,
            tracer=tracer,
        )

        if visualizer.plot_fit_no_hyper:

            fit = self.masked_imaging_fit_for_tracer(
                tracer=tracer,
                hyper_image_sky=None,
                hyper_background_noise=None,
                use_hyper_scalings=False,
            )

            visualizer.visualize_fit_imaging(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def make_attributes(self):
        return Attributes(
            cosmology=self.cosmology,
            positions=self.masked_dataset.positions,
            hyper_model_image=self.hyper_model_image,
            hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
        )

    def save_results_for_aggregator(
        self, paths: af.Paths, samples: af.OptimizerSamples
    ):

        if conf.instance["general"]["hyper"]["stochastic_outputs"]:
            self.save_stochastic_outputs(paths=paths, samples=samples)


class Attributes(AgAttributes):
    def __init__(
        self, cosmology, positions, hyper_model_image, hyper_galaxy_image_path_dict
    ):
        super().__init__(
            cosmology=cosmology,
            hyper_model_image=hyper_model_image,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
        )

        self.positions = positions
