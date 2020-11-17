from autoarray.structures import grids
from autogalaxy.galaxy import galaxy as g
from autogalaxy.pipeline.phase.abstract import result
from autolens.lens import positions_solver as pos


class Result(result.Result):
    @property
    def max_log_likelihood_plane(self):
        raise NotImplementedError()

    @property
    def max_log_likelihood_tracer(self):

        instance = self.analysis.associate_hyper_images(instance=self.instance)

        return self.analysis.tracer_for_instance(instance=instance)

    @property
    def source_plane_light_profile_centres(self) -> grids.GridCoordinates:
        """Return a list of all light profiles centres of all galaxies in the most-likely tracer's source-plane.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions."""
        return self.max_log_likelihood_tracer.source_plane.light_profile_centres

    @property
    def source_plane_inversion_centres(self) -> grids.GridCoordinates:
        """Return a list of all centres of a pixelized source reconstruction in the source-plane of the most likely fit.
        The brightest source pixel(s) are used to determine these centres.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions."""
        try:
            return (
                self.max_log_likelihood_fit.inversion.brightest_reconstruction_pixel_centre
            )
        except AttributeError:
            return []

    @property
    def source_plane_centres(self) -> grids.GridCoordinates:
        """Combine the source-plane light profile and inversion centres (see above) into a single list of source-plane
        centres.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model."""

        centres = list(self.source_plane_light_profile_centres) + list(
            self.source_plane_inversion_centres
        )

        return grids.GridCoordinates(coordinates=centres)

    @property
    def image_plane_multiple_image_positions_of_source_plane_centres(
        self,
    ) -> grids.GridCoordinates:
        """Backwards ray-trace the source-plane centres (see above) to the image-plane via the mass model, to determine
        the multiple image position of the source(s) in the image-plane..

        These image-plane positions are used by the next phase in a pipeline if automatic position updating is turned
        on."""

        # TODO : In the future, the multiple image positions functioon wil use an in-built adaptive grid.

        grid = self.analysis.masked_dataset.mask.geometry.unmasked_grid_sub_1

        solver = pos.PositionsFinder(grid=grid, pixel_scale_precision=0.001)

        try:
            multiple_images = [
                solver.solve(
                    lensing_obj=self.max_log_likelihood_tracer,
                    source_plane_coordinate=centre,
                )
                for centre in self.source_plane_centres.in_list[0]
            ]
            return grids.GridCoordinates(coordinates=multiple_images)
        except IndexError:
            return None

    @property
    def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return self.instance.path_instance_tuples_for_class(cls=g.Galaxy)
