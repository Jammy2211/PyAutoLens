from astropy import cosmology as cosmo

import autofit as af
from autolens.lens import ray_tracing
from autolens.model.galaxy import galaxy as g


class AbstractPhase(af.AbstractPhase):
    def __init__(
        self,
        phase_name,
        phase_tag=None,
        phase_folders=tuple(),
        optimizer_class=af.MultiNest,
        cosmology=cosmo.Planck15,
        auto_link_priors=False,
    ):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit
        models and hyper_galaxies passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        phase_name: str
            The name of this phase
        """

        super().__init__(
            phase_name=phase_name,
            phase_tag=phase_tag,
            phase_folders=phase_folders,
            optimizer_class=optimizer_class,
            auto_link_priors=auto_link_priors,
        )

        self.cosmology = cosmology

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

    def customize_priors(self, results):
        """
        Perform any prior or constant passing. This could involve setting model
        attributes equal to priors or constants from a previous phase.

        Parameters
        ----------
        results: autofit.tools.pipeline.ResultsCollection
            The result of the previous phase
        """
        pass

    # noinspection PyAbstractClass
    class Analysis(af.Analysis):
        def __init__(self, cosmology, results=None):
            """
            An lens object

            Parameters
            ----------
            results: autofit.tools.pipeline.ResultsCollection
                The results of all previous phases
            """

            self.results = results
            self.cosmology = cosmology

        @property
        def last_results(self):
            """
            Returns
            -------
            result: AbstractPhase.Result | None
                The result from the last phase
            """
            if self.results is not None:
                return self.results.last

        def tracer_for_instance(self, instance):
            return ray_tracing.Tracer.from_galaxies(
                galaxies=instance.galaxies, cosmology=self.cosmology
            )

    def make_result(self, result, analysis):
        return self.__class__.Result(
            constant=result.constant,
            figure_of_merit=result.figure_of_merit,
            previous_variable=result.previous_variable,
            gaussian_tuples=result.gaussian_tuples,
            analysis=analysis,
            optimizer=self.optimizer,
        )

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

    def run(self, image, results=None, mask=None):
        raise NotImplementedError()
