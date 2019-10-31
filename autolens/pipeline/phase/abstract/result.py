import autofit as af
from autoastro.galaxy import galaxy as g


class Result(af.Result):
    def __init__(
        self,
        constant,
        figure_of_merit,
        previous_variable,
        gaussian_tuples,
        analysis,
        optimizer,
    ):
        """
        The result of a phase
        """
        super().__init__(
            constant=constant,
            figure_of_merit=figure_of_merit,
            previous_variable=previous_variable,
            gaussian_tuples=gaussian_tuples,
        )

        self.analysis = analysis
        self.optimizer = optimizer

    @property
    def most_likely_tracer(self):
        return self.analysis.tracer_for_instance(instance=self.constant)

    @property
    def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return self.constant.path_instance_tuples_for_class(cls=g.Galaxy)
