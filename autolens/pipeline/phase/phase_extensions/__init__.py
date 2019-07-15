import copy

import numpy as np
from typing import cast

import autofit as af
from autolens import exc
from autolens.lens import lens_data as ld, lens_fit
from autolens.model.galaxy import galaxy as g
from autolens.model.hyper import hyper_data as hd
from autolens.model.inversion import pixelizations as px
from autolens.model.inversion import regularization as rg
from autolens.pipeline.phase import phase as ph
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline.phase.phase import setup_phase_mask
from autolens.pipeline.plotters import hyper_plotters
from .hyper_galaxy_phase import HyperGalaxyPhase
from .hyper_phase import HyperPhase




# noinspection PyAbstractClass
class VariableFixingHyperPhase(HyperPhase):

    def __init__(
            self,
            phase: ph.Phase,
            hyper_name: str,
            variable_classes=tuple(),
            default_classes=None
    ):
        super().__init__(
            phase=phase,
            hyper_name=hyper_name
        )
        self.default_classes = default_classes or dict()
        self.variable_classes = variable_classes

    def make_hyper_phase(self):
        phase = super().make_hyper_phase()

        phase.const_efficiency_mode = af.conf.instance.non_linear.get(
            'MultiNest',
            'extension_inversion_const_efficiency_mode',
            bool
        )
        phase.optimizer.sampling_efficiency = af.conf.instance.non_linear.get(
            'MultiNest',
            'extension_inversion_sampling_efficiency',
            float
        )
        phase.optimizer.n_live_points = af.conf.instance.non_linear.get(
            'MultiNest',
            'extension_inversion_n_live_points',
            int
        )

        return phase

    def run_hyper(self, data, results=None, **kwargs):
        """
        Run the phase, overriding the optimizer's variable instance with one created to
        only fit pixelization hyperparameters.
        """

        variable = copy.deepcopy(results.last.variable)
        self.transfer_classes(results.last.constant, variable)
        self.add_defaults(variable)

        phase = self.make_hyper_phase()
        phase.optimizer.variable = variable

        return phase.run(data, results=results, **kwargs)

    def add_defaults(self, variable: af.ModelMapper):
        """
        Add default prior models for each of the items in the defaults dictionary.

        Provides a way of specifying new prior models to be included at the top level
        in this phase.

        Parameters
        ----------
        variable
            The variable object to be used in this phase to which default prior
            models are attached.
        """
        for key, value in self.default_classes.items():
            if not hasattr(variable, key):
                setattr(variable, key, value)

    def transfer_classes(self, instance, mapper):
        """
        Recursively overwrite priors in the mapper with constant values from the
        instance except where the containing class is the descendant of a listed class.

        Parameters
        ----------
        instance
            The best fit from the previous phase
        mapper
            The prior variable from the previous phase
        """
        for key, instance_value in instance.__dict__.items():
            try:
                mapper_value = getattr(mapper, key)
                if isinstance(mapper_value, af.Prior):
                    setattr(mapper, key, instance_value)
                if not any(
                        isinstance(
                            instance_value,
                            cls
                        )
                        for cls in self.variable_classes
                ):
                    try:
                        self.transfer_classes(
                            instance_value,
                            mapper_value)
                    except AttributeError:
                        setattr(mapper, key, instance_value)
            except AttributeError:
                pass


class InversionPhase(VariableFixingHyperPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(
            self,
            phase: ph.Phase,
            variable_classes=(
                    px.Pixelization,
                    rg.Regularization
            ),
            default_classes=None
    ):
        super().__init__(
            phase=phase,
            variable_classes=variable_classes,
            hyper_name="inversion",
            default_classes=default_classes
        )

    @property
    def uses_inversion(self):
        return True

    @property
    def uses_hyper_images(self):
        return True


class InversionBackgroundSkyPhase(InversionPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: ph.Phase):
        super().__init__(
            phase=phase,
            variable_classes=(
                px.Pixelization,
                rg.Regularization,
                hd.HyperImageSky
            ),
            default_classes={
                "hyper_image_sky": hd.HyperImageSky
            }
        )


class InversionBackgroundNoisePhase(InversionPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: ph.Phase):
        super().__init__(
            phase=phase,
            variable_classes=(
                px.Pixelization,
                rg.Regularization,
                hd.HyperNoiseBackground
            ),
            default_classes={
                "hyper_noise_background": hd.HyperNoiseBackground
            }
        )


class InversionBackgroundBothPhase(InversionPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: ph.Phase):
        super().__init__(
            phase=phase,
            variable_classes=(
                px.Pixelization,
                rg.Regularization,
                hd.HyperImageSky,
                hd.HyperNoiseBackground
            ),
            default_classes={
                "hyper_image_sky": hd.HyperImageSky,
                "hyper_noise_background": hd.HyperNoiseBackground
            }
        )



class CombinedHyperPhase(HyperPhase):
    def __init__(
            self,
            phase: phase_imaging.PhaseImaging,
            hyper_phase_classes: (type,) = tuple()
    ):
        """
        A combined hyper phase that can run zero or more other hyper phases after the initial phase is run.

        Parameters
        ----------
        phase : phase_imaging.PhaseImaging
            The phase wrapped by this hyper phase
        hyper_phase_classes
            The classes of hyper phases to be run following the initial phase
        """
        super().__init__(
            phase,
            "combined"
        )
        self.hyper_phases = list(map(
            lambda cls: cls(phase),
            hyper_phase_classes
        ))

    @property
    def phase_names(self) -> [str]:
        """
        The names of phases included in this combined phase
        """
        return [
            phase.hyper_name
            for phase
            in self.hyper_phases
        ]

    def run(self, data, results: af.ResultsCollection = None, **kwargs) -> af.Result:
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
        results = copy.deepcopy(results) if results is not None else af.ResultsCollection()
        result = self.phase.run(data, results=results, **kwargs)
        results.add(self.phase.phase_name, result)

        for hyper_phase in self.hyper_phases:
            hyper_result = hyper_phase.run_hyper(
                data=data,
                results=results,
                **kwargs
            )
            setattr(result, hyper_phase.hyper_name, hyper_result)

        setattr(
            result,
            self.hyper_name,
            self.run_hyper(
                data,
                results
            )
        )
        return result

    def combine_variables(self, result) -> af.ModelMapper:
        """
        Combine the variable objects from all previous results in this combined hyper phase.

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
        variable = self.combine_variables(
            results.last
        )

        phase = self.make_hyper_phase()
        phase.optimizer.phase_tag = ''
        phase.optimizer.variable = variable

        phase.phase_tag = ''

        return phase.run(data, results=results, **kwargs)
