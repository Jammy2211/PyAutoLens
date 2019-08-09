import autofit as af
from autolens.model.hyper import hyper_data as hd
from autolens.model.inversion import pixelizations as px
from autolens.model.inversion import regularization as rg
from autolens.pipeline.phase import phase_imaging as ph
from .hyper_phase import HyperPhase


# noinspection PyAbstractClass
class VariableFixingHyperPhase(HyperPhase):
    def __init__(
        self, phase: ph.PhaseImaging, hyper_name: str, variable_classes=tuple()
    ):
        super().__init__(phase=phase, hyper_name=hyper_name)
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

    def make_variable(self, constant):
        return constant.as_variable(self.variable_classes)

    def run_hyper(self, data, results=None, **kwargs):
        """
        Run the phase, overriding the optimizer's variable instance with one created to
        only fit pixelization hyperparameters.
        """
        phase = self.make_hyper_phase()
        phase.optimizer.variable = self.make_variable(results.last.constant)

        return phase.run(
            data,
            results=results,
            mask=results.last.mask_2d,
            positions=results.last.positions,
        )


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
    ):
        super().__init__(
            phase=phase, variable_classes=variable_classes, hyper_name="inversion"
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
                hd.HyperBackgroundNoise,
            ),
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
                hd.HyperBackgroundNoise,
            ),
        )
