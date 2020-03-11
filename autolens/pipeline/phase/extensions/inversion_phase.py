import autofit as af
from autoastro.hyper import hyper_data as hd
from autoarray.operators.inversion import pixelizations as pix
from autoarray.operators.inversion import regularization as reg
from autolens.pipeline.phase import abstract
from autolens.pipeline.phase.imaging.phase import PhaseImaging
from .hyper_phase import HyperPhase


# noinspection PyAbstractClass
class ModelFixingHyperPhase(HyperPhase):
    def __init__(
        self, phase: abstract.AbstractPhase, hyper_name: str, model_classes=tuple()
    ):
        super().__init__(phase=phase, hyper_name=hyper_name)
        self.model_classes = model_classes

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
        phase.optimizer.evidence_tolerance = af.conf.instance.non_linear.get(
            "MultiNest", "extension_inversion_evidence_tolerance", float
        )
        phase.optimizer.terminate_at_acceptance_ratio = af.conf.instance.non_linear.get(
            "MultiNest", "extension_inversion_terminate_at_acceptance_ratio", bool
        )
        phase.optimizer.acceptance_ratio_threshold = af.conf.instance.non_linear.get(
            "MultiNest", "extension_inversion_acceptance_ratio_threshold", float
        )

        return phase

    def make_model(self, instance):
        return instance.as_model(self.model_classes)

    def run_hyper(self, dataset, results=None, **kwargs):
        """
        Run the phase, overriding the optimizer's model instance with one created to
        only fit pixelization hyperparameters.
        """
        phase = self.make_hyper_phase()
        phase.model = self.make_model(results.last.instance)

        return phase.run(
            dataset,
            mask=results.last.mask,
            results=results,
            positions=results.last.positions,
        )


class InversionPhase(ModelFixingHyperPhase):
    """
    Phase that makes everything in the model from the previous phase equal to the
    corresponding value from the best fit except for models associated with
    pixelization
    """

    def __init__(
        self,
        phase: abstract.AbstractPhase,
        model_classes=(pix.Pixelization, reg.Regularization),
    ):
        super().__init__(
            phase=phase, model_classes=model_classes, hyper_name="inversion"
        )


class InversionBackgroundSkyPhase(InversionPhase):
    """
    Phase that makes everything in the model from the previous phase equal to the
    corresponding value from the best fit except for models associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging):
        super().__init__(
            phase=phase,
            model_classes=(pix.Pixelization, reg.Regularization, hd.HyperImageSky),
        )


class InversionBackgroundNoisePhase(InversionPhase):
    """
    Phase that makes everything in the model from the previous phase equal to the
    corresponding value from the best fit except for models associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging):
        super().__init__(
            phase=phase,
            model_classes=(
                pix.Pixelization,
                reg.Regularization,
                hd.HyperBackgroundNoise,
            ),
        )


class InversionBackgroundBothPhase(InversionPhase):
    """
    Phase that makes everything in the model from the previous phase equal to the
    corresponding value from the best fit except for models associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging):
        super().__init__(
            phase=phase,
            model_classes=(
                pix.Pixelization,
                reg.Regularization,
                hd.HyperImageSky,
                hd.HyperBackgroundNoise,
            ),
        )
