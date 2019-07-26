import autofit as af
from autolens.model.hyper import hyper_data as hd
from autolens.model.inversion import pixelizations as px
from autolens.model.inversion import regularization as rg
from autolens.pipeline.phase import phase_imaging as ph
from .hyper_phase import HyperPhase


# noinspection PyAbstractClass
class VariableFixingHyperPhase(HyperPhase):
    def __init__(
        self,
        phase: ph.PhaseImaging,
        hyper_name: str,
        variable_classes=tuple(),
        default_classes=None,
    ):
        super().__init__(phase=phase, hyper_name=hyper_name)
        self.default_classes = default_classes or dict()
        self.variable_classes = variable_classes

    def make_hyper_phase(self):

        phase = super().make_hyper_phase()

        phase.optimizer.const_efficiency_mode = af.conf.instance.non_linear.get(
            "MultiNest", "extension_inversion_const_efficiency_mode", bool
        )
        phase.optimizer.sampling_efficiency = af.conf.instance.non_linear.get(
            "MultiNest", "extension_inversion_sampling_efficiency", float
        )
        phase.optimizer.n_live_points = af.conf.instance.non_linear.get(
            "MultiNest", "extension_inversion_n_live_points", int
        )

        return phase

    def run_hyper(self, data, results=None, **kwargs):
        """
        Run the phase, overriding the optimizer's variable instance with one created to
        only fit pixelization hyperparameters.
        """

        variable = results.last.variable.copy_with_fixed_priors(
            results.last.constant, self.variable_classes
        )
        self.add_defaults(variable)

        phase = self.make_hyper_phase()
        phase.optimizer.variable = variable

        return phase.run(
            data,
            results=results,
            mask=results.last.mask_2d,
            positions=results.last.positions,
        )

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
            if not hasattr(variable, key) or getattr(variable, key) is None:
                setattr(variable, key, value)


class InversionPhase(VariableFixingHyperPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(
        self,
        phase: ph.PhaseImaging,
        variable_classes=(px.Pixelization, rg.Regularization),
        default_classes=None,
    ):
        super().__init__(
            phase=phase,
            variable_classes=variable_classes,
            hyper_name="inversion",
            default_classes=default_classes,
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

    def __init__(self, phase: ph.PhaseImaging):
        super().__init__(
            phase=phase,
            variable_classes=(px.Pixelization, rg.Regularization, hd.HyperImageSky),
            default_classes={"hyper_image_sky": hd.HyperImageSky},
        )


class InversionBackgroundNoisePhase(InversionPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: ph.PhaseImaging):
        super().__init__(
            phase=phase,
            variable_classes=(
                px.Pixelization,
                rg.Regularization,
                hd.HyperNoiseBackground,
            ),
            default_classes={"hyper_noise_background": hd.HyperNoiseBackground},
        )


class InversionBackgroundBothPhase(InversionPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: ph.PhaseImaging):
        super().__init__(
            phase=phase,
            variable_classes=(
                px.Pixelization,
                rg.Regularization,
                hd.HyperImageSky,
                hd.HyperNoiseBackground,
            ),
            default_classes={
                "hyper_image_sky": hd.HyperImageSky,
                "hyper_noise_background": hd.HyperNoiseBackground,
            },
        )
