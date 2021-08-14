from astropy import cosmology as cosmo
import copy
import json
import numpy as np
from os import path
from scipy.stats import norm
from typing import List

import autofit as af
import autoarray as aa

from autogalaxy.analysis.analysis import AnalysisDataset as AgAnalysisDataset

from autolens import exc
from autolens.analysis.visualizer import Visualizer
from autolens.lens.ray_tracing import Tracer
from autolens.lens.settings import SettingsLens


class AnalysisLensing:
    def __init__(self, settings_lens=SettingsLens(), cosmology=cosmo.Planck15):

        self.cosmology = cosmology
        self.settings_lens = settings_lens

    def tracer_for_instance(self, instance):

        if hasattr(instance, "perturbation"):
            instance.galaxies.subhalo = instance.perturbation

        return Tracer.from_galaxies(
            galaxies=instance.galaxies, cosmology=self.cosmology
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
        preloads=aa.Preloads(),
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
            preloads=preloads,
        )

        AnalysisLensing.__init__(
            self=self, settings_lens=settings_lens, cosmology=cosmology
        )

        self.positions = positions

        self.settings_lens = settings_lens

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