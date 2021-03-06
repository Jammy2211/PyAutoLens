import pickle
import json
import autofit as af
from autogalaxy.pipeline.phase import abstract
from autogalaxy.pipeline.phase import extensions
from os import path
import os
import math
from scipy.stats import norm
import numpy as np

# noinspection PyAbstractClass
class StochasticPhase(extensions.HyperPhase):
    def __init__(
        self,
        phase: abstract.AbstractPhase,
        hyper_search,
        model_classes=tuple(),
        stochastic_method="gaussian",
        stochastic_sigma=0.0,
        subhalo_centre_width=None,
        subhalo_mass_at_200_log_uniform=True,
    ):

        self.is_stochastic = True
        self.stochastic_method = stochastic_method
        self.stochastic_sigma = stochastic_sigma

        super().__init__(
            phase=phase, hyper_search=hyper_search, model_classes=model_classes
        )

        self.subhalo_centre_width = subhalo_centre_width
        self.subhalo_mass_at_200_log_uniform = subhalo_mass_at_200_log_uniform

    @property
    def hyper_name(self):
        return "stochastic"

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

        # TODO : This is a horror show. It will be cleaner once autofit doesnt use .zip files anymore.

        self.results = results

        self.search.paths.restore()

        stochastic_log_evidences_json_file = path.join(
            self.paths.output_path, "stochastic_log_evidences.json"
        )

        try:
            with open(stochastic_log_evidences_json_file, "r") as f:
                stochastic_log_evidences = np.asarray(json.load(f))
        except FileNotFoundError:
            stochastic_log_evidences = results.last.stochastic_log_evidences

        self.search.paths.zip_remove()

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

            stochastic_tag = self.stochastic_method

        phase = self.make_hyper_phase()
        phase.hyper_name = f"{phase.hyper_name}_{stochastic_tag}"

        phase.settings.log_likelihood_cap = log_likelihood_cap

        phase.use_as_hyper_dataset = False

        phase.model = self.make_model(instance=results.last.instance)
        phase.model.galaxies.lens.take_attributes(
            source=results.last.model.galaxies.lens
        )
        if hasattr(phase.model.galaxies, "subhalo"):

            phase.model.galaxies.subhalo.take_attributes(
                source=results.last.model.galaxies.subhalo
            )

            if self.subhalo_centre_width is not None:
                phase.model.galaxies.subhalo.mass.centre = results.last.model_absolute(
                    a=self.subhalo_centre_width
                ).galaxies.subhalo.mass.centre

            if self.subhalo_mass_at_200_log_uniform:
                phase.model.galaxies.subhalo.mass.mass_at_200 = af.LogUniformPrior(
                    lower_limit=1e6, upper_limit=1e11
                )

        # TODO : Nasty hack to get log evidnees to copy, do something bettter in future.

        phase.modify_search_paths()

        phase.search.paths.restore()

        try:
            os.makedirs(self.paths.output_path)
        except FileExistsError:
            pass

        stochastic_log_evidences_json_file = path.join(
            phase.paths.output_path, "stochastic_log_evidences.json"
        )

        try:
            with open(stochastic_log_evidences_json_file, "w") as outfile:
                json.dump(
                    [float(evidence) for evidence in stochastic_log_evidences], outfile
                )
        except FileExistsError:
            pass

        stochastic_log_evidences_pickle_file = path.join(
            phase.paths.pickle_path, "stochastic_log_evidences.pickle"
        )

        with open(stochastic_log_evidences_pickle_file, "wb") as f:
            pickle.dump(stochastic_log_evidences, f)

        phase.search.paths.zip_remove()

        return phase.run(
            dataset,
            mask=results.last.mask,
            results=results,
            info=info,
            pickle_files=pickle_files,
            log_likelihood_cap=log_likelihood_cap,
        )
