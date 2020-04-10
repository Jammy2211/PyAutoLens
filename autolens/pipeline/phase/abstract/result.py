import autofit as af
from autoarray.structures import grids
from autoastro.galaxy import galaxy as g


class Result(af.Result):
    def __init__(
        self,
        instance,
        likelihood,
        previous_model,
        gaussian_tuples,
        analysis,
        optimizer,
        use_as_hyper_dataset=False,
    ):
        """
        The result of a phase
        """
        super().__init__(
            instance=instance,
            likelihood=likelihood,
            previous_model=previous_model,
            gaussian_tuples=gaussian_tuples,
        )

        self.analysis = analysis
        self.optimizer = optimizer
        self.use_as_hyper_dataset = use_as_hyper_dataset

    @property
    def most_likely_tracer(self):
        return self.analysis.tracer_for_instance(instance=self.instance)

    @property
    def source_plane_light_profile_centres(self) -> grids.Coordinates:
        """Return a list of all light profiles centres of all galaxies in the most-likely tracer's source-plane.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions."""
        return self.most_likely_tracer.source_plane.light_profile_centres

    @property
    def source_plane_inversion_centres(self) -> grids.Coordinates:
        """Return a list of all centres of a pixelized source reconstruction in the source-plane of the most likely fit.
        The brightest source pixel(s) are used to determine these centres.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions."""
        try:
            return self.most_likely_fit.inversion.brightest_reconstruction_pixel_centre
        except AttributeError:
            return grids.Coordinates(coordinates=[])

    @property
    def source_plane_centres(self) -> grids.Coordinates:
        """Combine the source-plane light profile and inversion centres (see above) into a single list of source-plane
        centres.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model."""
        centres = (
            self.source_plane_light_profile_centres
            + self.source_plane_inversion_centres
        )
        return grids.Coordinates(coordinates=centres)

    @property
    def image_plane_multiple_image_positions_of_source_plane_centres(
        self
    ) -> grids.Coordinates:
        """Backwards ray-trace the source-plane centres (see above) to the image-plane via the mass model, to determine
        the multiple image position of the source(s) in the image-plane..

        These image-plane positions are used by the next phase in a pipeline if automatic position updating is turned
        on."""

        # TODO : In the future, the multiple image positions functioon wil use an in-built adaptive grid.

        grid = self.analysis.masked_dataset.mask.geometry.unmasked_grid

        # TODO: Tracer method will ultimately return Coordinates, need to determine best way to implement method.

        positions = list(
            map(
                lambda centre: self.most_likely_tracer.image_plane_multiple_image_positions(
                    grid=grid, source_plane_coordinate=centre
                )[
                    0
                ],
                self.source_plane_centres,
            )
        )

        return grids.Coordinates(coordinates=positions)

    @property
    def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return self.instance.path_instance_tuples_for_class(cls=g.Galaxy)
