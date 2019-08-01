import copy

import autofit as af
from autolens.pipeline.phase import phase_imaging
from .hyper_galaxy_phase import HyperGalaxyPhase
from .hyper_phase import HyperPhase
from .inversion_phase import InversionBackgroundBothPhase
from .inversion_phase import InversionBackgroundNoisePhase
from .inversion_phase import InversionBackgroundSkyPhase
from .inversion_phase import InversionPhase
from .inversion_phase import VariableFixingHyperPhase


class CombinedHyperPhase(HyperPhase):
    def __init__(
        self, phase: phase_imaging.PhaseImaging, hyper_phase_classes: (type,) = tuple()
    ):
        """
        A hyper_combined hyper phase that can run zero or more other hyper phases after the initial phase is run.

        Parameters
        ----------
        phase : phase_imaging.PhaseImaging
            The phase wrapped by this hyper phase
        hyper_phase_classes
            The classes of hyper phases to be run following the initial phase
        """
        super().__init__(phase, "hyper_combined")
        self.hyper_phases = list(map(lambda cls: cls(phase), hyper_phase_classes))

    @property
    def phase_names(self) -> [str]:
        """
        The names of phases included in this hyper_combined phase
        """
        return [phase.hyper_name for phase in self.hyper_phases]

    def run(
        self,
        data,
        results: af.ResultsCollection = None,
        mask=None,
        positions=None,
        **kwargs
    ) -> af.Result:
        """
        Run the regular phase followed by the hyper phases. Each result of a hyper phase is attached to the
        overall result object by the hyper_name of that phase.

        Finally, a phase in run with all of the variable results from all the individual hyper phases.

        Parameters
        ----------
        data
            The data
        results
            Results from previous phases
        kwargs

        Returns
        -------
        result
            The result of the regular phase, with hyper results attached by associated hyper names
        """

        results = (
            copy.deepcopy(results) if results is not None else af.ResultsCollection()
        )
        result = self.phase.run(
            data, results=results, mask=mask, positions=positions, **kwargs
        )
        results.add(self.phase.phase_name, result)

        for phase in self.hyper_phases:
            hyper_result = phase.run_hyper(data=data, results=results, **kwargs)
            setattr(result, phase.hyper_name, hyper_result)

        setattr(result, self.hyper_name, self.run_hyper(data=data, results=results))
        return result

    def combine_variables(self, result) -> af.ModelMapper:
        """
        Combine the variable objects from all previous results in this hyper_combined hyper phase.

        Iterates through the hyper names of the included hyper phases, extracting a result
        for each name and adding the variable of that result to a new variable.

        Parameters
        ----------
        result
            The last result (with attribute results associated with phases in this phase)

        Returns
        -------
        combined_variable
            A variable object including all variables from results in this phase.
        """
        variable = af.ModelMapper()
        for name in self.phase_names:
            variable += getattr(result, name).variable
        return variable

    def run_hyper(self, data, results, **kwargs) -> af.Result:
        variable = self.combine_variables(results.last)

        phase = self.make_hyper_phase()
        phase.optimizer.phase_tag = ""
        phase.optimizer.variable = variable

        phase.phase_tag = ""

        return phase.run(
            data,
            results=results,
            mask=results.last.mask_2d,
            positions=results.last.positions,
            **kwargs
        )
