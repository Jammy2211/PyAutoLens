from astropy import cosmology as cosmo
import copy
import json
import logging
import numpy as np
import os
from os import path
from scipy.stats import norm
from typing import Dict, Optional, List

import autofit as af
import autoarray as aa

from autogalaxy.analysis.analysis import AnalysisDataset as AgAnalysisDataset

from autolens.lens.model.preloads import Preloads

from autolens import exc
from autolens.lens.model.maker import FitMaker
from autolens.lens.model.visualizer import Visualizer
from autolens.lens.ray_tracing import Tracer
from autolens.lens.model.settings import SettingsLens

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisLensing:
    def __init__(self, settings_lens=SettingsLens(), cosmology=cosmo.Planck15):

        self.cosmology = cosmology
        self.settings_lens = settings_lens

    def tracer_for_instance(self, instance, profiling_dict: Optional[Dict] = None):

        if hasattr(instance, "perturbation"):
            instance.galaxies.subhalo = instance.perturbation

        return Tracer.from_galaxies(
            galaxies=instance.galaxies,
            cosmology=self.cosmology,
            profiling_dict=profiling_dict,
        )


class AnalysisDataset(AgAnalysisDataset, AnalysisLensing):
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
        """

        Parameters
        ----------
        dataset
        positions : aa.Grid2DIrregular
            Image-pixel coordinates in arc-seconds of bright regions of the lensed source that will map close to one
            another in the source-plane(s) for an accurate mass model, which can be used to discard unphysical mass
            models during model-fitting.
        cosmology
        settings_pixelization
        settings_inversion
        settings_lens
        preloads
        """

        super().__init__(
            dataset=dataset,
            hyper_dataset_result=hyper_dataset_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
        )

        AnalysisLensing.__init__(
            self=self, settings_lens=settings_lens, cosmology=cosmology
        )

        self.positions = positions

        self.settings_lens = settings_lens

        self.preloads = Preloads()

    def set_preloads(self, paths: af.DirectoryPaths, model: af.Collection):

        try:
            os.makedirs(paths.profile_path)
        except FileExistsError:
            pass

        fit_maker = FitMaker(model=model, fit_func=self.fit_func)

        fit_0 = fit_maker.fit_via_model(unit_value=0.45)
        fit_1 = fit_maker.fit_via_model(unit_value=0.55)

        if fit_0 is None or fit_1 is None:
            self.preloads = Preloads(failed=True)
        else:
            self.preloads = Preloads.setup_all_via_fits(fit_0=fit_0, fit_1=fit_1)
            self.preloads.check_via_fit(fit=fit_0)

        self.preloads.output_info_to_summary(file_path=paths.profile_path)

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

    def modify_after_fit(
        self, paths: af.DirectoryPaths, model: af.AbstractPriorModel, result: af.Result
    ):

        self.output_or_check_figure_of_merit_sanity(paths=paths, result=result)
        self.preloads.reset_all()

        return self

    def log_likelihood_cap_from(self, stochastic_log_evidences_json_file):

        try:
            with open(stochastic_log_evidences_json_file, "r") as f:
                stochastic_log_evidences = np.asarray(json.load(f))
        except FileNotFoundError:
            raise exc.AnalysisException(
                "The file 'stochastic_log_evidences.json' could not be found in the output of the model-fitting results"
                "in the analysis before the stochastic analysis. Rerun PyAutoLens with `stochastic_outputs=True` in the"
                "`general.ini` configuration file."
            )

        mean, sigma = norm.fit(stochastic_log_evidences)

        return mean

    def stochastic_log_evidences_for_instance(self, instance) -> List[float]:
        raise NotImplementedError()

    def save_settings(self, paths: af.DirectoryPaths):

        super().save_settings(paths=paths)

        paths.save_object("settings_lens", self.settings_lens)

    def save_stochastic_outputs(
        self, paths: af.DirectoryPaths, samples: af.OptimizerSamples
    ):

        stochastic_log_evidences_json_file = path.join(
            paths.output_path, "stochastic_log_evidences.json"
        )

        try:
            with open(stochastic_log_evidences_json_file, "r") as f:
                stochastic_log_evidences = np.asarray(json.load(f))
        except FileNotFoundError:
            instance = samples.max_log_likelihood_instance
            stochastic_log_evidences = self.stochastic_log_evidences_for_instance(
                instance=instance
            )

        if stochastic_log_evidences is None:
            return

        with open(stochastic_log_evidences_json_file, "w") as outfile:
            json.dump(
                [float(evidence) for evidence in stochastic_log_evidences], outfile
            )

        paths.save_object("stochastic_log_evidences", stochastic_log_evidences)

        visualizer = Visualizer(visualize_path=paths.image_path)

        visualizer.visualize_stochastic_histogram(
            log_evidences=stochastic_log_evidences,
            max_log_evidence=np.max(samples.log_likelihood_list),
            histogram_bins=self.settings_lens.stochastic_histogram_bins,
        )

    @property
    def no_positions(self):

        # settings_lens = SettingsLens(
        # positions_threshold=None,
        # stochastic_likelihood_resamples=self.settings_lens.stochastic_likelihood_resamples,
        # stochastic_samples = self.settings_lens.stochastic_samples,
        # stochastic_histogram_bins = self.settings_lens.stochastic_histogram_bins
        # )
        #
        # return self.__class__(
        #     dataset=self.dataset,
        #     positions = None,
        #     hyper_dataset_result=self.hyper_result,
        #     cosmology=self.cosmology,
        #     settings_pixelization=self.settings_pixelization,
        #     settings_inversion=self.settings_inversion,
        #     settings_lens=settings_lens,
        #     preloads=self.preloads
        # )

        analysis = copy.deepcopy(self)

        analysis.positions = None
        analysis.settings_lens.positions_threshold = None

        return analysis

    @property
    def fit_func(self):
        raise NotImplementedError
