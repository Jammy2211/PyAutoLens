from os import path
import numpy as np
import json

from autoconf import conf
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures.grids.two_d import grid_2d_irregular
from autoarray.structures import visibilities
from autogalaxy.profiles import light_profiles as lp
from autogalaxy.galaxy import galaxy as g
from autogalaxy.analysis import result as res
from autolens.fit import fit_point_source
from autolens.lens import ray_tracing, positions_solver as pos


class Result(res.Result):
    @property
    def max_log_likelihood_tracer(self) -> ray_tracing.Tracer:

        return self.analysis.tracer_for_instance(instance=self.instance)

    @property
    def source_plane_light_profile_centre(self) -> grid_2d_irregular.Grid2DIrregular:
        """
        Return a light profile centres of a galaxy in the most-likely tracer's source-plane. If there are multiple
        light profiles, the first light profile's centre is returned.

        These centres are used by automatic position updating to determine the best-fit lens model's image-plane
        multiple-image positions.
        """
        centre = self.max_log_likelihood_tracer.source_plane.extract_attribute(
            cls=lp.LightProfile, attr_name="centre"
        )
        if centre is not None:
            return grid_2d_irregular.Grid2DIrregular(grid=[np.asarray(centre[0])])

    @property
    def source_plane_centre(self) -> grid_2d_irregular.Grid2DIrregular:
        """
        Return the centre of a source-plane galaxy via the following criteria:

        1) If the source plane contains only light profiles, return the first light's centre.
        2) If it contains an `Inversion` return the centre of its brightest pixel instead.

        These centres are used by automatic position updating to determine the multiple-images of a best-fit lens model
        (and thus tracer) by back-tracing the centres to the image plane via the mass model.
        """
        return self.source_plane_light_profile_centre

    @property
    def image_plane_multiple_image_positions(
        self,
    ) -> grid_2d_irregular.Grid2DIrregular:
        """
        Backwards ray-trace the source-plane centres (see above) to the image-plane via the mass model, to determine
        the multiple image position of the source(s) in the image-plane.

        These image-plane positions are used by the next search in a pipeline if automatic position updating is turned
        on."""

        # TODO : In the future, the multiple image positions functioon wil use an in-built adaptive grid.

        grid = self.analysis.dataset.mask.unmasked_grid_sub_1

        solver = pos.PositionsSolver(grid=grid, pixel_scale_precision=0.001)

        multiple_images = solver.solve(
            lensing_obj=self.max_log_likelihood_tracer,
            source_plane_coordinate=self.source_plane_centre.in_list[0],
        )

        return grid_2d_irregular.Grid2DIrregular(grid=multiple_images)

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
        factor : float
            The value the computed threshold is multipled by to make the position threshold larger or smaller than the
            maximum log likelihood model's threshold.
        minimum_threshold : float
            The output threshold is rounded up to this value if it is below it, to avoid extremely small threshold
            values.

        Returns
        -------
        float
            The maximum source plane separation of this results maximum likelihood `Tracer` multiple images multipled
            by `factor` and rounded up to the `threshold`.
        """

        positions = self.image_plane_multiple_image_positions

        positions_fits = fit_point_source.FitPositionsSourceMaxSeparation(
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
    def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return self.instance.path_instance_tuples_for_class(cls=g.Galaxy)


class ResultDataset(Result):
    @property
    def max_log_likelihood_tracer(self) -> ray_tracing.Tracer:

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
    def max_log_likelihood_pixelization_grids_of_planes(self):
        return self.max_log_likelihood_tracer.sparse_image_plane_grids_of_planes_from_grid(
            grid=self.analysis.dataset.grid
        )

    def image_for_galaxy(self, galaxy: g.Galaxy) -> np.ndarray:
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
    def image_galaxy_dict(self) -> {str: g.Galaxy}:
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

        hyper_model_image = array_2d.Array2D.manual_mask(
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

        try:
            with open(stochastic_log_evidences_json_file, "r") as f:
                return np.asarray(json.load(f))
        except FileNotFoundError:
            self.analysis.save_stochastic_outputs(
                paths=self.search.paths, samples=self.samples
            )
            with open(stochastic_log_evidences_json_file, "r") as f:
                return np.asarray(json.load(f))


class ResultImaging(ResultDataset):
    @property
    def max_log_likelihood_fit(self):

        hyper_image_sky = self.analysis.hyper_image_sky_for_instance(
            instance=self.instance
        )

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=self.instance
        )

        return self.analysis.fit_imaging_for_tracer(
            tracer=self.max_log_likelihood_tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    @property
    def unmasked_model_image(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image

    @property
    def unmasked_model_image_of_planes(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image_of_planes

    @property
    def unmasked_model_image_of_planes_and_galaxies(self):
        fit = self.max_log_likelihood_fit
        return fit.unmasked_blurred_image_of_planes_and_galaxies


class ResultInterferometer(ResultDataset):
    @property
    def max_log_likelihood_fit(self):

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=self.instance
        )

        return self.analysis.fit_interferometer_for_tracer(
            tracer=self.max_log_likelihood_tracer,
            hyper_background_noise=hyper_background_noise,
        )

    @property
    def real_space_mask(self):
        return self.max_log_likelihood_fit.interferometer.real_space_mask

    @property
    def unmasked_model_visibilities(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image

    @property
    def unmasked_model_visibilities_of_planes(self):
        return self.max_log_likelihood_fit.unmasked_blurred_image_of_planes

    @property
    def unmasked_model_visibilities_of_planes_and_galaxies(self):
        fit = self.max_log_likelihood_fit
        return fit.unmasked_blurred_image_of_planes_and_galaxies

    def visibilities_for_galaxy(self, galaxy: g.Galaxy) -> np.ndarray:
        """
        Parameters
        ----------
        galaxy
            A galaxy used in this search

        Returns
        -------
        ndarray or None
            A numpy arrays giving the model visibilities of that galaxy
        """
        return self.max_log_likelihood_fit.galaxy_model_visibilities_dict[galaxy]

    @property
    def visibilities_galaxy_dict(self) -> {str: g.Galaxy}:
        """
        A dictionary associating galaxy names with model visibilities of those galaxies
        """
        return {
            galaxy_path: self.visibilities_for_galaxy(galaxy)
            for galaxy_path, galaxy in self.path_galaxy_tuples
        }

    @property
    def hyper_galaxy_visibilities_path_dict(self):
        """
        A dictionary associating 1D hyper_galaxies galaxy visibilities with their names.
        """

        hyper_galaxy_visibilities_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:
            hyper_galaxy_visibilities_path_dict[path] = self.visibilities_galaxy_dict[
                path
            ]

        return hyper_galaxy_visibilities_path_dict

    @property
    def hyper_model_visibilities(self):

        hyper_model_visibilities = visibilities.Visibilities.zeros(
            shape_slim=(self.max_log_likelihood_fit.visibilities.shape_slim,)
        )

        for path, galaxy in self.path_galaxy_tuples:
            hyper_model_visibilities += self.hyper_galaxy_visibilities_path_dict[path]

        return hyper_model_visibilities


class ResultPoint(Result):
    @property
    def grid(self):
        return grid_2d.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

    @property
    def max_log_likelihood_fit(self):

        return self.analysis.fit_positions_for(tracer=self.max_log_likelihood_tracer)
