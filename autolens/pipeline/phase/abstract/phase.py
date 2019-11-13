import autofit as af
from autolens.pipeline.phase.abstract.result import Result


# noinspection PyAbstractClass


class AbstractPhase(af.AbstractPhase):
    Result = Result

    @af.convert_paths
    def __init__(self, paths, *, optimizer_class=af.MultiNest):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit
        models and hyper_galaxies passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        """

        super().__init__(paths=paths, optimizer_class=optimizer_class)

    @property
    def phase_folders(self):
        return self.optimizer.phase_folders

    @property
    def phase_property_collections(self):
        """
        Returns
        -------
        phase_property_collections: [PhaseProperty]
            A list of phase property collections associated with this phase. This is
            used in automated prior passing and should be overridden for any phase that
            contains its own PhasePropertys.
        """
        return []

    @property
    def path(self):
        return self.optimizer.path

    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            figure_of_merit=result.figure_of_merit,
            previous_model=result.previous_model,
            gaussian_tuples=result.gaussian_tuples,
            analysis=analysis,
            optimizer=self.optimizer,
        )

    def run(self, dataset, results=None, mask=None, positions=None):
        raise NotImplementedError()
