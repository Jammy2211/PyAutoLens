import copy

import autofit as af
from autolens.pipeline.phase import phase_imaging as ph


class HyperPhase(object):
    def __init__(self, phase: ph.PhaseImaging, hyper_name: str):
        """
        Abstract HyperPhase. Wraps a regular phase, performing that phase before performing the action
        specified by the run_hyper.

        Parameters
        ----------
        phase
            A regular phase
        """
        self.phase = phase
        self.hyper_name = hyper_name

    def run_hyper(self, *args, **kwargs) -> af.Result:
        """
        Run the hyper phase.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        result
            The result of the hyper phase.
        """
        raise NotImplementedError()

    def make_hyper_phase(self) -> ph.PhaseImaging:
        """
        Returns
        -------
        hyper_phase
            A copy of the original phase with a modified name and path
        """

        phase = copy.deepcopy(self.phase)

        phase_folders = phase.phase_folders
        phase_folders.append(phase.phase_name)

        phase.optimizer = phase.optimizer.copy_with_name_extension(
            extension=self.hyper_name + "_" + phase.phase_tag
        )

        # TODO : This addeds the methods to the combined phase, assuming they'll be overwritten for other
        # TODO : phases in their make_hyper_phase methods

        phase.optimizer.const_efficiency_mode = af.conf.instance.non_linear.get(
            "MultiNest", "extension_combined_const_efficiency_mode", bool
        )
        phase.optimizer.sampling_efficiency = af.conf.instance.non_linear.get(
            "MultiNest", "extension_combined_sampling_efficiency", float
        )
        phase.optimizer.n_live_points = af.conf.instance.non_linear.get(
            "MultiNest", "extension_combined_n_live_points", int
        )

        phase.optimizer.phase_tag = ""
        phase.phase_tag = ""
        phase.pass_priors = self.pass_priors
        phase.preload_pixelization_grid = None

        return phase

    def pass_priors(self, results):

        pass

    def run(self, data, results: af.ResultsCollection = None, **kwargs) -> af.Result:
        """
        Run the normal phase and then the hyper phase.

        Parameters
        ----------
        data
            Data
        results
            Results from previous phases.
        kwargs

        Returns
        -------
        result
            The result of the phase, with a hyper result attached as an attribute with the hyper_name of this
            phase.
        """

        results = (
            copy.deepcopy(results) if results is not None else af.ResultsCollection()
        )

        result = self.phase.run(data, results=results, **kwargs)
        results.add(self.phase.phase_name, result)
        hyper_result = self.run_hyper(data=data, results=results, **kwargs)
        setattr(result, self.hyper_name, hyper_result)
        return result

    def __getattr__(self, item):
        return getattr(self.phase, item)
