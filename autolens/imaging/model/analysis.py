import json
import logging
import numpy as np
import os
import time
from typing import Dict, Optional
from os import path

from autoconf import conf
import autofit as af
import autoarray as aa
import autogalaxy as ag

from autolens.lens.model.analysis import AnalysisDataset
from autolens.lens.model.preloads import Preloads
from autolens.imaging.model.maker import FitImagingMaker
from autolens.imaging.model.result import ResultImaging
from autolens.imaging.model.visualizer import VisualizerImaging
from autolens.imaging.fit_imaging import FitImaging

from autolens import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisImaging(AnalysisDataset):
    @property
    def imaging(self):
        return self.dataset

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.AbstractPriorModel):

        self.check_and_replace_hyper_images(paths=paths)

        if not paths.is_complete:

            visualizer = VisualizerImaging(visualize_path=paths.image_path)

            visualizer.visualize_imaging(imaging=self.imaging)

            visualizer.visualize_hyper_images(
                hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
                hyper_model_image=self.hyper_model_image,
            )

            logger.info(
                "PRELOADS - Setting up preloads, may take a few minutes for fits using an inversion."
            )

            self.set_preloads(paths=paths, model=model)

        return self

    def check_and_replace_hyper_images(self, paths: af.DirectoryPaths):

        try:
            hyper_model_image = paths.load_object("hyper_model_image")

            if np.max(abs(hyper_model_image - self.hyper_model_image)) > 1e-8:

                logger.info(
                    "ANALYSIS - Hyper image loaded from pickle different to that set in Analysis class."
                    "Overwriting hyper images with values loaded from pickles."
                )

                self.hyper_model_image = hyper_model_image

                hyper_galaxy_image_path_dict = paths.load_object(
                    "hyper_galaxy_image_path_dict"
                )
                self.hyper_galaxy_image_path_dict = hyper_galaxy_image_path_dict

        except (FileNotFoundError, AttributeError):
            pass

    def set_preloads(self, paths: af.DirectoryPaths, model: af.AbstractPriorModel):

        try:
            os.makedirs(paths.profile_path)
        except FileExistsError:
            pass

        fit_maker = FitImagingMaker(
            model=model, fit_from_instance_func=self.fit_imaging_for_instance
        )

        self.preloads = Preloads.setup_all_from_fit_maker(fit_maker=fit_maker)

        try:
            self.check_preloads(model=model)
        except exc.PreloadsException:
            self.preloads.use_w_tilde = False
            self.check_preloads(model=model)

        self.preloads.output_info_to_summary(file_path=paths.profile_path)

    def check_preloads(self, model: af.AbstractPriorModel):

        fit_maker = FitImagingMaker(
            model=model, fit_from_instance_func=self.fit_imaging_for_instance
        )

        if conf.instance["general"]["test"]["check_preloads"]:
            self.preloads.check_via_fit_maker(fit_maker=fit_maker)

    def modify_after_fit(
        self, paths: af.DirectoryPaths, model: af.AbstractPriorModel, result: af.Result
    ):

        self.output_or_check_figure_of_merit_sanity(paths=paths, result=result)
        self.preloads.reset_all()

        return self

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
        self,
        instance,
        use_hyper_scalings=True,
        preload_overwrite=None,
        check_positions=True,
        profiling_dict: Optional[Dict] = None,
    ):

        self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(
            instance=instance, profiling_dict=profiling_dict
        )

        if check_positions:
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
            profiling_dict=profiling_dict,
        )

    def fit_imaging_for_tracer(
        self,
        tracer,
        hyper_image_sky,
        hyper_background_noise,
        use_hyper_scalings=True,
        preload_overwrite=None,
        profiling_dict: Optional[Dict] = None,
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
            profiling_dict=profiling_dict,
        )

    def profile_log_likelihood_function(
        self, instance, paths: Optional[af.DirectoryPaths] = None
    ):

        profiling_dict = {}
        info_dict = {}

        repeats = conf.instance["general"]["profiling"]["repeats"]
        info_dict["repeats"] = repeats

        start = time.time()

        for i in range(repeats):
            fit = self.fit_imaging_for_instance(instance=instance)
            fit.figure_of_merit

        fit_time = (time.time() - start) / repeats

        info_dict["fit_time"] = fit_time

        fit = self.fit_imaging_for_instance(
            instance=instance, profiling_dict=profiling_dict
        )
        fit.figure_of_merit

        info_dict["image_pixels"] = self.imaging.grid.sub_shape_slim
        info_dict["sub_size_light_profiles"] = self.imaging.grid.sub_size
        info_dict["sub_size_inversion"] = self.imaging.grid_inversion.sub_size
        info_dict["psf_shape_2d"] = self.imaging.psf.shape_native

        if fit.inversion is not None:
            info_dict["source_pixels"] = len(fit.inversion.reconstruction)

        if hasattr(fit.inversion, "w_tilde"):
            info_dict[
                "w_tilde_curvature_preload_size"
            ] = fit.inversion.w_tilde.curvature_preload.shape[0]
            info_dict["snr_cut"] = fit.inversion.w_tilde.snr_cut

        if paths is not None:

            try:
                os.makedirs(paths.profile_path)
            except FileExistsError:
                pass

            with open(path.join(paths.profile_path, "profiling_dict.json"), "w+") as f:
                json.dump(fit.profiling_dict, f, indent=4)

            with open(path.join(paths.profile_path, "info_dict.json"), "w+") as f:
                json.dump(info_dict, f, indent=4)

        return profiling_dict

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

        visualizer.visualize_fit_imaging(fit=fit, during_analysis=during_analysis)
        visualizer.visualize_tracer(
            tracer=fit.tracer, grid=fit.grid, during_analysis=during_analysis
        )
        if fit.inversion is not None:
            visualizer.visualize_inversion(
                inversion=fit.inversion, during_analysis=during_analysis
            )

        visualizer.visualize_contribution_maps(tracer=fit.tracer)

        if visualizer.plot_fit_no_hyper:
            fit = self.fit_imaging_for_tracer(
                tracer=fit.tracer,
                hyper_image_sky=None,
                hyper_background_noise=None,
                use_hyper_scalings=False,
                preload_overwrite=Preloads(use_w_tilde=False),
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
        if self.preloads.sparse_image_plane_grids_of_planes is not None:
            paths.save_object(
                "preload_sparse_grids_of_planes",
                self.preloads.sparse_image_plane_grids_of_planes,
            )
