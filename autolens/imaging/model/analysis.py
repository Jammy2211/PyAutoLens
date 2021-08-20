from autoconf import conf
import autofit as af
import autoarray as aa
import autogalaxy as ag

from autolens.lens.model.analysis import AnalysisDataset
from autolens.lens.model.preloads import Preloads
from autolens.imaging.model.result import ResultImaging
from autolens.imaging.model.visualizer import VisualizerImaging
from autolens.imaging.fit_imaging import FitImaging

from autolens import exc


class AnalysisImaging(AnalysisDataset):
    @property
    def imaging(self):
        return self.dataset

    def modify_before_fit(self, model: af.AbstractPriorModel):

        self.set_preloads(model=model)
        self.check_preloads(model=model)

    def check_preloads(self, model: af.AbstractPriorModel):

        if not conf.instance["general"]["test"]["check_preloads"]:
            return
        instance = model.instance_from_prior_medians()

        fom_with_preloads = self.fit_imaging_for_instance(
            instance=instance
        ).figure_of_merit

        fom_without_preloads = self.fit_imaging_for_instance(
            instance=instance, preload_overwrite=Preloads(use_w_tilde=False)
        ).figure_of_merit

        if abs(fom_with_preloads - fom_without_preloads) > 1e-8:

            raise exc.AnalysisException(
                f"The log likelihood of fits using and not using preloads are not"
                f"consistent, indicating preloading has gone wrong."
                f"The likelihood values are {fom_with_preloads} (with preloads) and "
                f"{fom_without_preloads} (without preloads)"
            )

    def preload_fit_list_from_unit_values(self, model: af.AbstractPriorModel):

        instance_0 = model.instance_from_unit_vector(
            unit_vector=[0.45] * model.prior_count
        )
        instance_1 = model.instance_from_unit_vector(
            unit_vector=[0.55] * model.prior_count
        )

        fit_0 = self.fit_imaging_for_instance(
            instance=instance_0, preload_overwrite=Preloads(use_w_tilde=False)
        )
        fit_1 = self.fit_imaging_for_instance(
            instance=instance_1, preload_overwrite=Preloads(use_w_tilde=False)
        )

        return fit_0, fit_1

    def preload_fit_list_from_random_instances(self, model: af.AbstractPriorModel):

        preload_attempts = 100

        fit_list = []

        for i in range(preload_attempts):

            instance = model.random_instance()

            try:
                fit_list.append(
                    self.fit_imaging_for_instance(
                        instance=instance, preload_overwrite=Preloads(use_w_tilde=False)
                    )
                )
            except Exception:
                pass

            if len(fit_list) == 2:
                break

            if i == preload_attempts:
                raise exc.AnalysisException("Unable to set preloads.")

        return fit_list

    def set_preloads(self, model: af.AbstractPriorModel):

        fit_0, fit_1 = self.preload_fit_list_from_unit_values(model=model)

        self.preloads = Preloads()

        self.preloads.set_blurred_image(fit_0=fit_0, fit_1=fit_1)
        self.preloads.set_sparse_grid_of_planes(fit_0=fit_0, fit_1=fit_1)
        self.preloads.set_mapper(fit_0=fit_0, fit_1=fit_1)
        self.preloads.set_inversion(fit_0=fit_0, fit_1=fit_1)
        self.preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)

    def log_likelihood_function(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the imaging in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model imaging itself
        """

        try:
            return self.fit_imaging_for_instance(instance=instance).figure_of_merit
        except (
            exc.PixelizationException,
            exc.InversionException,
            exc.GridException,
            OverflowError,
        ) as e:
            raise exc.FitException from e

    def fit_imaging_for_instance(
        self, instance, use_hyper_scalings=True, preload_overwrite=None
    ):

        self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        self.settings_lens.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer, positions=self.positions
        )

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        return self.fit_imaging_for_tracer(
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scalings=use_hyper_scalings,
            preload_overwrite=preload_overwrite,
        )

    def fit_imaging_for_tracer(
        self,
        tracer,
        hyper_image_sky,
        hyper_background_noise,
        use_hyper_scalings=True,
        preload_overwrite=None,
    ):

        preloads = self.preloads if preload_overwrite is None else preload_overwrite

        return FitImaging(
            imaging=self.dataset,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scaling=use_hyper_scalings,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            preloads=preloads,
        )

    def stochastic_log_evidences_for_instance(self, instance):

        instance = self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        if not tracer.has_pixelization:
            return

        if not isinstance(
            tracer.pixelizations_of_planes[-1], aa.pix.VoronoiBrightnessImage
        ):
            return

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        settings_pixelization = (
            self.settings_pixelization.settings_with_is_stochastic_true()
        )

        log_evidences = []

        for i in range(self.settings_lens.stochastic_samples):

            try:
                log_evidence = FitImaging(
                    imaging=self.dataset,
                    tracer=tracer,
                    hyper_image_sky=hyper_image_sky,
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

        fit = self.fit_imaging_for_instance(instance=instance)

        visualizer = VisualizerImaging(visualize_path=paths.image_path)

        visualizer.visualize_imaging(imaging=self.imaging)
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
            tracer=fit.tracer,
        )

        if visualizer.plot_fit_no_hyper:
            fit = self.fit_imaging_for_tracer(
                tracer=fit.tracer,
                hyper_image_sky=None,
                hyper_background_noise=None,
                use_hyper_scalings=False,
            )

            visualizer.visualize_fit_imaging(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def save_results_for_aggregator(
        self,
        paths: af.DirectoryPaths,
        samples: af.OptimizerSamples,
        model: af.Collection,
    ):

        pixelization = ag.util.model.pixelization_from(model=model)

        if conf.instance["general"]["hyper"]["stochastic_outputs"]:
            if ag.util.model.isinstance_or_prior(
                pixelization, aa.pix.VoronoiBrightnessImage
            ):
                self.save_stochastic_outputs(paths=paths, samples=samples)

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return ResultImaging(samples=samples, model=model, analysis=self, search=search)

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):

        super().save_attributes_for_aggregator(paths=paths)

        paths.save_object("psf", self.dataset.psf_unormalized)
        paths.save_object("mask", self.dataset.mask)
        paths.save_object("positions", self.positions)
        if self.preloads.sparse_grids_of_planes is not None:
            paths.save_object(
                "preload_sparse_grids_of_planes", self.preloads.sparse_grids_of_planes
            )
