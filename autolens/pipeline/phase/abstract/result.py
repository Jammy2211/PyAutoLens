import autofit as af
from autoarray.structures import grids
from autoastro.galaxy import galaxy as g
from autolens.fit import fit


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
    def source_plane_light_profile_centres(self):
        """Return a list of all light profiles centres in the most-likely tracer's source-plane.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model."""
        if self.most_likely_tracer.source_plane.has_light_profile:
            return self.most_likely_tracer.light_profile_centres_of_planes[-1]
        else:
            return []

    @property
    def source_plane_inversion_centres(self):
        """Return a list of the centres of a pixelized source reconstruction in the source-plane of a most likely fit.
        The brightest source pixel(s) are used to determine these centres.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model."""
        if self.most_likely_tracer.source_plane.has_pixelization:
            return [
                self.most_likely_fit.inversion.brightest_reconstruction_pixel_centre
            ]
        else:
            return []

    @property
    def source_plane_centres(self):
        """Combine the source-plane light profile and inversion centres (see above) into a single list of source-plane
        centres.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model."""
        centres = (
            self.source_plane_light_profile_centres
            + self.source_plane_inversion_centres
        )
        return grids.Coordinates(coordinates=[centres])

    @property
    def image_plane_multiple_image_positions_of_source_plane_centres(self):
        """Combine the source-plane light profile and inversion centres (see above) into a single list of source-plane
        centres.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model."""

        # TODO : In the future, the multiple image positions functioon wil use an in-built adaptive grid.

        grid = self.analysis.masked_dataset.mask.geometry.unmasked_grid

        return list(
            map(
                lambda centre: self.most_likely_tracer.image_plane_multiple_image_positions(
                    grid=grid, source_plane_coordinate=centre
                ),
                self.source_plane_centres[0],
            )
        )

    @property
    def image_plane_multiple_image_position_source_plane_separations(self):

        positions_fits = list(
            map(
                lambda positions: fit.FitPositions(
                    positions=positions,
                    tracer=self.most_likely_tracer,
                    noise_map=self.analysis.masked_dataset.mask.pixel_scales,
                ),
                self.image_plane_multiple_image_positions_of_source_plane_centres,
            )
        )

        return list(map(lambda fit: fit.maximum_separations[0], positions_fits))

    @property
    def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return self.instance.path_instance_tuples_for_class(cls=g.Galaxy)
