import autofit as af
from autoastro.hyper import hyper_data as hd
from autoarray.operators.inversion import pixelizations as pix
from autoarray.operators.inversion import regularization as reg
from autolens.pipeline.phase import abstract
from autolens.pipeline.phase.imaging.phase import PhaseImaging
from .hyper_phase import HyperPhase


# noinspection PyAbstractClass
class VariableFixingHyperPhase(HyperPhase):
    def __init__(
        self, phase: abstract.AbstractPhase, hyper_name: str, variable_classes=tuple()
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
        phase.optimizer.multimodal = af.conf.instance.non_linear.get(
            "MultiNest", "extension_inversion_multimodal", bool
        )

        return phase

    def make_variable(self, constant):
        return constant.as_variable(self.variable_classes)

    def run_hyper(self, dataset, results=None, **kwargs):
        """
        Run the phase, overriding the optimizer's variable instance with one created to
        only fit pixelization hyperparameters.
        """
        phase = self.make_hyper_phase()
        phase.variable = self.make_variable(results.last.constant)

        return phase.run(
            dataset,
            results=results,
            mask=results.last.mask,
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
        phase: abstract.AbstractPhase,
        variable_classes=(pix.Pixelization, reg.Regularization),
    ):
        super().__init__(
            phase=phase, variable_classes=variable_classes, hyper_name="inversion"
        )


class InversionBackgroundSkyPhase(InversionPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging):
        super().__init__(
            phase=phase,
            variable_classes=(pix.Pixelization, reg.Regularization, hd.HyperImageSky),
        )


class InversionBackgroundNoisePhase(InversionPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging):
        super().__init__(
            phase=phase,
            variable_classes=(
                pix.Pixelization,
                reg.Regularization,
                hd.HyperBackgroundNoise,
            ),
        )


class InversionBackgroundBothPhase(InversionPhase):
    """
    Phase that makes everything in the variable from the previous phase equal to the
    corresponding value from the best fit except for variables associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging):
        super().__init__(
            phase=phase,
            variable_classes=(
                pix.Pixelization,
                reg.Regularization,
                hd.HyperImageSky,
                hd.HyperBackgroundNoise,
            ),
        )
