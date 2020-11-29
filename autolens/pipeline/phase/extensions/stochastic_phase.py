from autoconf import conf
import autofit as af
from autogalaxy.pipeline.phase import abstract
from autogalaxy.pipeline.phase import extensions

from os import path
import math
from scipy.stats import norm
import numpy as np

# noinspection PyAbstractClass
class StochasticPhase(extensions.ModelFixingHyperPhase):
    def __init__(
        self,
        phase: abstract.AbstractPhase,
        hyper_search,
        model_classes=tuple(),
        stochastic_method="gaussian",
        stochastic_sigma=0.0,
    ):

        self.is_stochastic = True
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

        try:
            stochastic_log_evidences = self.load_stochastic_log_evidences_from_json()
        except FileNotFoundError:
            stochastic_log_evidences = results.last.stochastic_log_evidences

        try:
            self.save_stochastic_log_evidences_to_json(
                stochastic_log_evidences=stochastic_log_evidences
            )
        except FileExistsError:
            pass

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

        # TODO : Nasty hack to get log evidnees to copy, do something bettter in future.

        phase.modify_search_paths()

        print(phase.stochastic_log_evidences_json_file)

        try:
            phase.save_stochastic_log_evidences_to_json(
                stochastic_log_evidences=stochastic_log_evidences
            )
        except FileExistsError:
            pass

        phase.save_stochastic_log_evidences_to_pickle(
            stochastic_log_evidences=stochastic_log_evidences
        )

        result = phase.run(
            dataset,
            mask=results.last.mask,
            results=results,
            info=info,
            pickle_files=pickle_files,
            log_likelihood_cap=log_likelihood_cap,
        )

        return result
