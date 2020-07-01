import autofit as af
from autogalaxy.pipeline.phase import abstract
from autogalaxy.pipeline.phase import extensions

import numpy as np

# noinspection PyAbstractClass
class StochasticPhase(extensions.ModelFixingHyperPhase):
    def __init__(self, phase: abstract.AbstractPhase, search, model_classes=tuple()):
        super().__init__(
            phase=phase,
            search=search,
            model_classes=model_classes,
            hyper_name="stochastic",
        )

    def make_model(self, instance):
        return instance.as_model(self.model_classes)

    def run_hyper(self, dataset, info=None, results=None, **kwargs):
        """
        Run the phase, overriding the search's model instance with one created to
        only fit pixelization hyperparameters.
        """

        self.results = results or af.ResultsCollection()

        log_evidences = results.last.stochastic_log_evidences

        phase = self.make_hyper_phase()
        phase.settings.log_likelihood_cap = np.median(log_evidences)
        phase.use_as_hyper_dataset = False
        phase.model = self.make_model(results.last.instance)

        return phase.run(dataset, mask=results.last.mask, results=results)
