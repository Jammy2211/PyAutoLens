import numpy as np

from os import path
import numpy as np
import json

from autoarray.structures.grids.two_d import grid_2d_irregular
from autogalaxy.profiles import light_profiles as lp
from autogalaxy.galaxy import galaxy as g
from autogalaxy.analysis import result as res
from autolens.lens import ray_tracing, positions_solver as pos


class Result(res.Result):
    @property
    def max_log_likelihood_tracer(self) -> ray_tracing.Tracer:

        instance = self.analysis.associate_hyper_images(instance=self.instance)

        return self.analysis.tracer_for_instance(instance=instance)

    @property
    def source_plane_light_profile_centre(self) -> grid_2d_irregular.Grid2DIrregular:
        """
        Return a light profile centres of a galaxy in the most-likely tracer's source-plane. If there are multiple
        light profiles, the first light profile's centre is returned.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions.
        """
        centre = self.max_log_likelihood_tracer.source_plane.extract_attribute(
            cls=lp.LightProfile, name="centre"
        )
        if centre is not None:
            return grid_2d_irregular.Grid2DIrregular(grid=[np.asarray(centre[0])])

    @property
    def source_plane_inversion_centre(self) -> grid_2d_irregular.Grid2DIrregular:
        """
        Returns the centre of the brightest source pixel(s) of an `Inversion`.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions.
        """
        if self.max_log_likelihood_fit.inversion is not None:
            return (
                self.max_log_likelihood_fit.inversion.brightest_reconstruction_pixel_centre
            )

    @property
    def source_plane_centre(self) -> grid_2d_irregular.Grid2DIrregular:
        """
        Return the centre of a source-plane galaxy via the following criteria:

        1) If the source plane contains only light profiles, return the first light's centre.
        2) If it contains an `Inversion` return the centre of its brightest pixel instead.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model.
        """
        if self.source_plane_inversion_centre is not None:
            return self.source_plane_inversion_centre
        elif self.source_plane_light_profile_centre is not None:
            return self.source_plane_light_profile_centre

    @property
    def image_plane_multiple_image_positions_of_source_plane_centres(
        self,
    ) -> grid_2d_irregular.Grid2DIrregular:
        """Backwards ray-trace the source-plane centres (see above) to the image-plane via the mass model, to determine
        the multiple image position of the source(s) in the image-plane..

        These image-plane positions are used by the next phase in a pipeline if automatic position updating is turned
        on."""

        # TODO : In the future, the multiple image positions functioon wil use an in-built adaptive grid.

        grid = self.analysis.masked_dataset.mask.unmasked_grid_sub_1

        solver = pos.PositionsSolver(grid=grid, pixel_scale_precision=0.001)

        try:

            multiple_images = solver.solve(
                lensing_obj=self.max_log_likelihood_tracer,
                source_plane_coordinate=self.source_plane_centre.in_list[0],
            )
            return grid_2d_irregular.Grid2DIrregular(grid=multiple_images)
        except (AttributeError, IndexError):
            return None

    @property
    def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return self.instance.path_instance_tuples_for_class(cls=g.Galaxy)


class ResultDataset(Result, res.ResultDataset):
    @property
    def mask(self):
        return self.max_log_likelihood_fit.mask

    @property
    def positions(self):
        return self.max_log_likelihood_fit.masked_dataset.positions

    @property
    def pixelization(self):
        for galaxy in self.max_log_likelihood_fit.tracer.galaxies:
            if galaxy.pixelization is not None:
                return galaxy.pixelization

    @property
    def max_log_likelihood_pixelization_grids_of_planes(self):
        return self.max_log_likelihood_tracer.sparse_image_plane_grids_of_planes_from_grid(
            grid=self.max_log_likelihood_fit.grid
        )

    @property
    def stochastic_log_evidences(self):

        stochastic_log_evidences_json_file = path.join(
            self.search.paths.output_path, "stochastic_log_evidences.json"
        )

        try:
            with open(stochastic_log_evidences_json_file, "r") as f:
                return np.asarray(json.load(f))
        except FileNotFoundError:
            pass
