from os import path
import numpy as np
import json

from autoconf import conf
import autoarray as aa
import autogalaxy as ag

from autogalaxy.analysis.result import Result as AgResult

from autolens.point.fit_point import FitPositionsSourceMaxSeparation
from autolens.lens.ray_tracing import Tracer
from autolens.point.point_solver import PointSolver


class Result(AgResult):
    @property
    def max_log_likelihood_tracer(self) -> Tracer:

        return self.analysis.tracer_for_instance(instance=self.instance)

    @property
    def source_plane_light_profile_centre(self) -> aa.Grid2DIrregular:
        """
        Return a light profile centres of a galaxy in the most-likely tracer's source-plane. If there are multiple
        light profiles, the first light profile's centre is returned.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions.
        """
        centre = self.max_log_likelihood_tracer.source_plane.extract_attribute(
            cls=ag.lp.LightProfile, attr_name="centre"
        )
        if centre is not None:
            return aa.Grid2DIrregular(grid=[np.asarray(centre[0])])

    @property
    def source_plane_centre(self) -> aa.Grid2DIrregular:
        """
        Return the centre of a source-plane galaxy via the following criteria:

        1) If the source plane contains only light profiles, return the first light's centre.
        2) If it contains an `LinearEqn` return the centre of its brightest pixel instead.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model.
        """
        return self.source_plane_light_profile_centre

    @property
    def image_plane_multiple_image_positions(self) -> aa.Grid2DIrregular:
        """
        Backwards ray-trace the source-plane centres (see above) to the image-plane via the mass model, to determine
        the multiple image position of the source(s) in the image-plane.

        These image-plane positions are used by the next search in a pipeline if automatic position updating is turned
        on."""

        # TODO : In the future, the multiple image positions functioon wil use an in-built adaptive grid.

        grid = self.analysis.dataset.mask.unmasked_grid_sub_1

        solver = PointSolver(grid=grid, pixel_scale_precision=0.001)

        multiple_images = solver.solve(
            lensing_obj=self.max_log_likelihood_tracer,
            source_plane_coordinate=self.source_plane_centre.in_list[0],
        )

        return aa.Grid2DIrregular(grid=multiple_images)

    def positions_threshold_from(self, factor=1.0, minimum_threshold=None) -> float:
        """
        Compute a new position threshold from these results corresponding to the image-plane multiple image positions of
         the maximum log likelihood `Tracer` ray-traced to the source-plane.

        First, we ray-trace forward the multiple-image's to the source-plane via the mass model to determine how far
        apart they are separated. We take the maximum source-plane separation of these points and multiple this by
        the auto_positions_factor to determine a new positions threshold. This value may also be rounded up to the
        input `auto_positions_minimum_threshold`.

        This is used for non-linear search chaining, specifically updating the position threshold of a new model-fit
        using the maximum likelihood model of a previous search.

        Parameters
        ----------
        factor
            The value the computed threshold is multipled by to make the position threshold larger or smaller than the
            maximum log likelihood model's threshold.
        minimum_threshold
            The output threshold is rounded up to this value if it is below it, to avoid extremely small threshold
            values.

        Returns
        -------
        float
            The maximum source plane separation of this results maximum likelihood `Tracer` multiple images multipled
            by `factor` and rounded up to the `threshold`.
        """

        positions = self.image_plane_multiple_image_positions

        positions_fits = FitPositionsSourceMaxSeparation(
            positions=positions, noise_map=None, tracer=self.max_log_likelihood_tracer
        )

        positions_threshold = factor * np.max(
            positions_fits.max_separation_of_source_plane_positions
        )

        if minimum_threshold is not None:
            if positions_threshold < minimum_threshold:
                return minimum_threshold

        return positions_threshold

    @property
    def path_galaxy_tuples(self) -> [(str, ag.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return self.instance.path_instance_tuples_for_class(cls=ag.Galaxy)


class ResultDataset(Result):
    @property
    def max_log_likelihood_tracer(self) -> Tracer:

        instance = self.analysis.associate_hyper_images(instance=self.instance)

        return self.analysis.tracer_for_instance(instance=instance)

    @property
    def max_log_likelihood_fit(self):
        raise NotImplementedError

    @property
    def mask(self):
        return self.analysis.dataset.mask

    @property
    def grid(self):
        return self.analysis.dataset.grid

    @property
    def positions(self):
        return self.analysis.positions

    @property
    def pixelization(self):
        for galaxy in self.max_log_likelihood_tracer.galaxies:
            if galaxy.pixelization is not None:
                return galaxy.pixelization

    @property
    def source_plane_centre(self) -> aa.Grid2DIrregular:
        """
        Return the centre of a source-plane galaxy via the following criteria:

        1) If the source plane contains only light profiles, return the first light's centre.
        2) If it contains an `LinearEqn` return the centre of its brightest pixel instead.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model.
        """
        if self.source_plane_inversion_centre is not None:
            return self.source_plane_inversion_centre
        elif self.source_plane_light_profile_centre is not None:
            return self.source_plane_light_profile_centre

    @property
    def source_plane_inversion_centre(self) -> aa.Grid2DIrregular:
        """
        Returns the centre of the brightest source pixel(s) of an `LinearEqn`.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions.
        """
        if self.max_log_likelihood_fit.inversion is not None:
            return self.max_log_likelihood_fit.inversion.brightest_reconstruction_pixel_centre_list[
                0
            ]

    @property
    def max_log_likelihood_sparse_image_plane_grid_list_of_planes(self):
        return self.max_log_likelihood_tracer.sparse_image_plane_grid_list_of_planes_from(
            grid=self.analysis.dataset.grid
        )

    def image_for_galaxy(self, galaxy: ag.Galaxy) -> np.ndarray:
        """
        Parameters
        ----------
        galaxy
            A galaxy used in this search

        Returns
        -------
        ndarray or None
            A numpy arrays giving the model image of that galaxy
        """
        return self.max_log_likelihood_fit.galaxy_model_image_dict[galaxy]

    @property
    def image_galaxy_dict(self) -> {str: ag.Galaxy}:
        """
        A dictionary associating galaxy names with model images of those galaxies
        """
        return {
            galaxy_path: self.image_for_galaxy(galaxy)
            for galaxy_path, galaxy in self.path_galaxy_tuples
        }

    @property
    def hyper_galaxy_image_path_dict(self):
        """
        A dictionary associating 1D hyper_galaxies galaxy images with their names.
        """

        hyper_minimum_percent = conf.instance["general"]["hyper"][
            "hyper_minimum_percent"
        ]

        hyper_galaxy_image_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:

            galaxy_image = self.image_galaxy_dict[path]

            if not np.all(galaxy_image == 0):
                minimum_galaxy_value = hyper_minimum_percent * max(galaxy_image)
                galaxy_image[galaxy_image < minimum_galaxy_value] = minimum_galaxy_value

            hyper_galaxy_image_path_dict[path] = galaxy_image

        return hyper_galaxy_image_path_dict

    @property
    def hyper_model_image(self):

        hyper_model_image = aa.Array2D.manual_mask(
            array=np.zeros(self.mask.mask_sub_1.pixels_in_mask),
            mask=self.mask.mask_sub_1,
        )

        for path, galaxy in self.path_galaxy_tuples:
            hyper_model_image += self.hyper_galaxy_image_path_dict[path]

        return hyper_model_image

    @property
    def stochastic_log_evidences(self):

        stochastic_log_evidences_json_file = path.join(
            self.search.paths.output_path, "stochastic_log_evidences.json"
        )

        self.search.paths.restore()

        try:
            with open(stochastic_log_evidences_json_file, "r") as f:
                stochastic_log_evidences = np.asarray(json.load(f))
        except FileNotFoundError:
            self.analysis.save_stochastic_outputs(
                paths=self.search.paths, samples=self.samples
            )
            with open(stochastic_log_evidences_json_file, "r") as f:
                stochastic_log_evidences = np.asarray(json.load(f))

        self.search.paths.zip_remove()

        return stochastic_log_evidences
