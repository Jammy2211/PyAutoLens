import autofit as af
from autoastro.galaxy import galaxy as g


class Result(af.Result):
    def __init__(
        self,
        instance,
        figure_of_merit,
        previous_model,
        gaussian_tuples,
        analysis,
        optimizer,
    ):
        """
        The result of a phase
        """
        super().__init__(
            instance=instance,
            figure_of_merit=figure_of_merit,
            previous_model=previous_model,
            gaussian_tuples=gaussian_tuples,
        )

        self.analysis = analysis
        self.optimizer = optimizer

    @property
    def most_likely_tracer(self):
        return self.analysis.tracer_for_instance(instance=self.instance)

    @property
    def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return self.instance.path_instance_tuples_for_class(cls=g.Galaxy)
