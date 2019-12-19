import copy

import autofit as af
from autofit.tools.phase import Dataset
from autolens.pipeline.phase import abstract


class HyperPhase(object):
    def __init__(self, phase: abstract.AbstractPhase, hyper_name: str):
        """
        Abstract HyperPhase. Wraps a phase, performing that phase before performing the action
        specified by the run_hyper.

        Parameters
        ----------
        phase
            A phase
        """
        self.phase = phase
        self.hyper_name = hyper_name

    def run_hyper(self, *args, **kwargs) -> af.Result:
        """
        Run the hyper_galaxies phase.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        result
            The result of the hyper_galaxies phase.
        """
        raise NotImplementedError()

    def make_hyper_phase(self) -> abstract.AbstractPhase:
        """
        Returns
        -------
        hyper_phase
            A copy of the original phase with a modified name and path
        """
        phase = copy.deepcopy(self.phase)
        phase.paths.zip()

        phase.optimizer = phase.optimizer.copy_with_name_extension(
            extension=self.hyper_name + "_" + phase.paths.phase_tag,
            remove_phase_tag=True,
        )

        phase.optimizer.const_efficiency_mode = af.conf.instance.non_linear.get(
            "MultiNest", "extension_combined_const_efficiency_mode", bool
        )
        phase.optimizer.sampling_efficiency = af.conf.instance.non_linear.get(
            "MultiNest", "extension_combined_sampling_efficiency", float
        )
        phase.optimizer.n_live_points = af.conf.instance.non_linear.get(
            "MultiNest", "extension_combined_n_live_points", int
        )
        phase.optimizer.multimodal = af.conf.instance.non_linear.get(
            "MultiNest", "extension_combined_multimodal", bool
        )

        phase.is_hyper_phase = True
        phase.customize_priors = self.customize_priors

        return phase

    def customize_priors(self, results):
        pass

    def run(
        self, dataset: Dataset, results: af.ResultsCollection = None, **kwargs
    ) -> af.Result:
        """
        Run the hyper phase and then the hyper_galaxies phase.

        Parameters
        ----------
        dataset
            Data
        results
            Results from previous phases.
        kwargs

        Returns
        -------
        result
            The result of the phase, with a hyper_galaxies result attached as an attribute with the hyper_name of this
            phase.
        """
        dataset.save(self.paths.phase_output_path)

        results = (
            copy.deepcopy(results) if results is not None else af.ResultsCollection()
        )

        result = self.phase.run(dataset, results=results, **kwargs)
        results.add(self.phase.paths.phase_name, result)
        hyper_result = self.run_hyper(dataset=dataset, results=results, **kwargs)
        setattr(result, self.hyper_name, hyper_result)
        return result

    def __getattr__(self, item):
        return getattr(self.phase, item)
