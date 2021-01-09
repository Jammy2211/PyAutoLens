import autofit as af
from autolens.lens import ray_tracing
from autolens.pipeline import visualizer as vis
from os import path
import pickle
from typing import List
import json
import numpy as np


class Analysis:
    def plane_for_instance(self, instance):
        raise NotImplementedError()

    def tracer_for_instance(self, instance):

        return ray_tracing.Tracer.from_galaxies(
            galaxies=instance.galaxies, cosmology=self.cosmology
        )

    def stochastic_log_evidences_for_instance(self, instance) -> List[float]:
        raise NotImplementedError()

    def save_stochastic_outputs(self, paths: af.Paths, samples: af.OptimizerSamples):

        stochastic_log_evidences_json_file = path.join(
            paths.output_path, "stochastic_log_evidences.json"
        )
        stochastic_log_evidences_pickle_file = path.join(
            paths.pickle_path, "stochastic_log_evidences.pickle"
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

        with open(stochastic_log_evidences_pickle_file, "wb") as f:
            pickle.dump(stochastic_log_evidences, f)

        visualizer = vis.Visualizer(visualize_path=paths.image_path)

        visualizer.visualize_stochastic_histogram(
            log_evidences=stochastic_log_evidences,
            max_log_evidence=np.max(samples.log_likelihoods),
            histogram_bins=self.settings.settings_lens.stochastic_histogram_bins,
        )
