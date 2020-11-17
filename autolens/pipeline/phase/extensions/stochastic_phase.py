from autoconf import conf
import autofit as af
from autogalaxy.pipeline.phase import abstract
from autogalaxy.pipeline.phase import extensions

from os import path
import math
from scipy.stats import norm
import pickle
import json
import numpy as np

# noinspection PyAbstractClass
class StochasticPhase(extensions.ModelFixingHyperPhase):
    def __init__(
        self,
        phase: abstract.AbstractPhase,
        hyper_search,
        model_classes=tuple(),
        histogram_samples=100,
        histogram_bins=10,
        stochastic_method="gaussian",
        stochastic_sigma=0.0,
    ):

        self.is_stochastic = True
        self.histogram_samples = histogram_samples
        self.histogram_bins = histogram_bins
        self.stochastic_method = stochastic_method
        self.stochastic_sigma = stochastic_sigma

        super().__init__(
            phase=phase,
            hyper_search=hyper_search,
            model_classes=model_classes,
            hyper_name="stochastic",
        )

    def make_model(self, instance):
        return instance.as_model(self.model_classes)

    def run_hyper(
        self,
        dataset,
        results: af.ResultsCollection,
        info=None,
        pickle_files=None,
        **kwargs,
    ):
        """
        Run the phase, overriding the search's model instance with one created to
        only fit pixelization hyperparameters.
        """

        self.results = results

        stochastic_log_evidences_file = path.join(
            conf.instance.output_path,
            self.paths.path_prefix,
            self.paths.name,
            "stochastic_log_evidences.json",
        )

        try:
            stochastic_log_evidences = self.stochastic_log_evidences_from_json(
                filename=stochastic_log_evidences_file
            )
        except FileNotFoundError:
            stochastic_log_evidences = results.last.stochastic_log_evidences(
                histogram_samples=self.histogram_samples
            )
            self.stochastic_log_evidences_to_json(
                filename=stochastic_log_evidences_file,
                stochastic_log_evidences=stochastic_log_evidences,
            )

        if self.stochastic_method in "gaussian":

            mean, sigma = norm.fit(stochastic_log_evidences)

            limit = math.erf(0.5 * np.abs(self.stochastic_sigma) * math.sqrt(2))

            if self.stochastic_sigma >= 0.0:
                log_likelihood_cap = mean + (sigma * limit)
            else:
                log_likelihood_cap = mean - (sigma * limit)

            stochastic_tag = f"{self.stochastic_method}_{str(self.stochastic_sigma)}"

        else:

            log_likelihood_cap = np.median(stochastic_log_evidences)

            stochastic_tag = f"{self.stochastic_method}"

        phase = self.make_hyper_phase()
        phase.hyper_name = f"{phase.hyper_name}_{stochastic_tag}"

        phase.settings.log_likelihood_cap = log_likelihood_cap
        #       phase.paths.tag = phase.settings.phase_tag_no_inversion

        phase.use_as_hyper_dataset = False

        phase.model = self.make_model(results.last.instance)

        # TODO : HACK

        from autogalaxy.profiles import mass_profiles as mp

        mass = af.PriorModel(mp.EllipticalPowerLaw)

        mass.centre = af.last[-1].model.galaxies.lens.mass.centre
        mass.elliptical_comps = af.last[-1].model.galaxies.lens.mass.elliptical_comps
        mass.einstein_radius = af.last[-1].model.galaxies.lens.mass.einstein_radius

        phase.model.galaxies.lens.mass = mass

        result = phase.run(
            dataset,
            mask=results.last.mask,
            results=results,
            info=info,
            pickle_files=pickle_files,
            log_likelihood_cap=log_likelihood_cap,
        )

        self.save_stochastic_log_evidences(
            stochastic_log_evidences=stochastic_log_evidences
        )

        return result

    def save_stochastic_log_evidences(self, stochastic_log_evidences):
        """
        Save the dataset associated with the phase
        """
        with open(
            path.join(self.paths.pickle_path, "stochastic_log_evidences.pickle"), "wb"
        ) as f:
            pickle.dump(stochastic_log_evidences, f)

    def stochastic_log_evidences_from_json(cls, filename):
        with open(filename, "r") as f:
            return np.asarray(json.load(f))

    def stochastic_log_evidences_to_json(self, filename, stochastic_log_evidences):
        """
        Save the dataset associated with the phase
        """
        with open(filename, "w") as outfile:
            json.dump(
                [float(evidence) for evidence in stochastic_log_evidences], outfile
            )
