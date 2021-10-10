from astropy import cosmology as cosmo
import logging

from autoconf import conf
import autofit as af
import autoarray as aa
import autogalaxy as ag

from autolens.lens.model.analysis import AnalysisDataset
from autolens.lens.model.preloads import Preloads
from autolens.interferometer.model.result import ResultInterferometer
from autolens.interferometer.model.visualizer import VisualizerInterferometer
from autolens.interferometer.fit_interferometer import FitInterferometer
from autolens.lens.model.settings import SettingsLens

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisInterferometer(AnalysisDataset):
    def __init__(
        self,
        dataset,
        positions: aa.Grid2DIrregular = None,
        hyper_dataset_result=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
        settings_lens=SettingsLens(),
    ):

        super().__init__(
            dataset=dataset,
            positions=positions,
            hyper_dataset_result=hyper_dataset_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            settings_lens=settings_lens,
        )

        if self.hyper_dataset_result is not None:

            self.set_hyper_dataset(result=self.hyper_dataset_result)

        else:

            self.hyper_galaxy_visibilities_path_dict = None
            self.hyper_model_visibilities = None

    @property
    def interferometer(self):
        return self.dataset

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.AbstractPriorModel):

        self.check_and_replace_hyper_images(paths=paths)

        if not paths.is_complete:

            visualizer = VisualizerInterferometer(visualize_path=paths.image_path)

            visualizer.visualize_interferometer(interferometer=self.interferometer)

            visualizer.visualize_hyper_images(
                hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
                hyper_model_image=self.hyper_model_image,
            )

            logger.info(
                "PRELOADS - Setting up preloads, may take a few minutes for fits using an inversion."
            )

            self.set_preloads(paths=paths, model=model)

        return self

    def set_hyper_dataset(self, result):

        super().set_hyper_dataset(result=result)

        self.hyper_model_visibilities = result.hyper_model_visibilities
        self.hyper_galaxy_visibilities_path_dict = (
            result.hyper_galaxy_visibilities_path_dict
        )

    def associate_hyper_visibilities(
        self, instance: af.ModelInstance
    ) -> af.ModelInstance:
        """
        Takes visibilities from the last result, if there is one, and associates them with galaxies in this search
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
           The input instance with visibilities associated with galaxies where possible.
        """
        if self.hyper_galaxy_visibilities_path_dict is not None:
            for galaxy_path, galaxy in instance.path_instance_tuples_for_class(
                ag.Galaxy
            ):
                if galaxy_path in self.hyper_galaxy_visibilities_path_dict:
                    galaxy.hyper_model_visibilities = self.hyper_model_visibilities
                    galaxy.hyper_galaxy_visibilities = self.hyper_galaxy_visibilities_path_dict[
                        galaxy_path
                    ]

        return instance

    def log_likelihood_function(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the interferometer in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model interferometer itself
        """

        try:
            return self.fit_interferometer_for_instance(
                instance=instance
            ).figure_of_merit
        except (
            exc.PixelizationException,
            exc.InversionException,
            exc.GridException,
            OverflowError,
        ) as e:
            raise exc.FitException from e

    def fit_interferometer_for_instance(
        self,
        instance,
        use_hyper_scalings=True,
        preload_overwrite=None,
        check_positions=True,
    ):

        self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        if check_positions:
            self.settings_lens.check_positions_trace_within_threshold_via_tracer(
                tracer=tracer, positions=self.positions
            )

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        return self.fit_interferometer_for_tracer(
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scalings=use_hyper_scalings,
        )

    def fit_interferometer_for_tracer(
        self,
        tracer,
        hyper_background_noise,
        use_hyper_scalings=True,
        preload_overwrite=None,
    ):

        preloads = self.preloads if preload_overwrite is None else preload_overwrite

        return FitInterferometer(
            interferometer=self.dataset,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scaling=use_hyper_scalings,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            preloads=preloads,
        )

    @property
    def fit_func(self):
        return self.fit_interferometer_for_instance

    def stochastic_log_evidences_for_instance(self, instance):

        instance = self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        if not tracer.has_pixelization:
            return None

        if not any(
            [
                isinstance(pix, aa.pix.VoronoiBrightnessImage)
                for pix in tracer.pixelization_list
            ]
        ):
            return

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        settings_pixelization = (
            self.settings_pixelization.settings_with_is_stochastic_true()
        )

        log_evidences = []

        for i in range(self.settings_lens.stochastic_samples):

            try:
                log_evidence = FitInterferometer(
                    interferometer=self.dataset,
                    tracer=tracer,
                    hyper_background_noise=hyper_background_noise,
                    settings_pixelization=settings_pixelization,
                    settings_inversion=self.settings_inversion,
                    preloads=self.preloads,
                ).log_evidence
            except (
                exc.PixelizationException,
                exc.InversionException,
                exc.GridException,
                OverflowError,
            ) as e:
                log_evidence = None

            if log_evidence is not None:
                log_evidences.append(log_evidence)

        return log_evidences

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis):

        instance = self.associate_hyper_images(instance=instance)

        fit = self.fit_interferometer_for_instance(instance=instance)

        visualizer = VisualizerInterferometer(visualize_path=paths.image_path)

        visualizer.visualize_fit_interferometer(
            fit=fit, during_analysis=during_analysis
        )
        visualizer.visualize_tracer(
            tracer=fit.tracer, grid=fit.grid, during_analysis=during_analysis
        )
        if fit.inversion is not None:
            visualizer.visualize_inversion(
                inversion=fit.inversion, during_analysis=during_analysis
            )

        visualizer.visualize_contribution_maps(tracer=fit.tracer)

        if visualizer.plot_fit_no_hyper:
            fit = self.fit_interferometer_for_tracer(
                tracer=fit.tracer,
                hyper_background_noise=None,
                use_hyper_scalings=False,
                preload_overwrite=Preloads(use_w_tilde=False),
            )

            visualizer.visualize_fit_interferometer(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def save_results_for_aggregator(
        self,
        paths: af.DirectoryPaths,
        samples: af.OptimizerSamples,
        model: af.Collection,
    ):

        if conf.instance["general"]["hyper"]["stochastic_outputs"]:
            self.save_stochastic_outputs(paths=paths, samples=samples)

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return ResultInterferometer(
            samples=samples, model=model, analysis=self, search=search
        )

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):

        super().save_attributes_for_aggregator(paths=paths)

        paths.save_object("uv_wavelengths", self.dataset.uv_wavelengths)
        paths.save_object("real_space_mask", self.dataset.real_space_mask)
        paths.save_object("positions", self.positions)
