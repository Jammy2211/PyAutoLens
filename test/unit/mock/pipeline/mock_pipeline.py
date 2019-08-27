import numpy as np

import autofit as af
from autolens.model.galaxy import galaxy as g
from autolens.array.util import binning_util


class MockAnalysis(object):
    def __init__(self, number_galaxies, value):
        self.number_galaxies = number_galaxies
        self.value = value

    # noinspection PyUnusedLocal
    def galaxy_images_for_model(self, model):
        return self.number_galaxies * [np.array([self.value])]

    def fit(self, instance):
        return 1


class MockResults(object):
    def __init__(
        self,
        model_image=None,
        mask=None,
        galaxy_images=(),
        constant=None,
        analysis=None,
        optimizer=None,
        pixelization=None,
    ):
        self.model_image = model_image
        self.unmasked_model_image = model_image
        self.mask_2d = mask
        self.galaxy_images = galaxy_images
        self.constant = constant or af.ModelInstance()
        self.variable = af.ModelMapper()
        self.analysis = analysis
        self.optimizer = optimizer
        self.pixelization = pixelization
        self.hyper_combined = MockHyperCombinedPhase()

    @property
    def path_galaxy_tuples(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return [(("g0",), g.Galaxy(redshift=0.5)), (("g1",), g.Galaxy(redshift=1.0))]

    @property
    def path_galaxy_tuples_with_index(self) -> [(str, g.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return [
            (0, ("g0",), g.Galaxy(redshift=0.5)),
            (1, ("g1",), g.Galaxy(redshift=1.0)),
        ]

    @property
    def image_2d_dict(self) -> {str: g.Galaxy}:
        """
        A dictionary associating galaxy names with model images of those galaxies
        """
        return {
            galaxy_path: self.galaxy_images[i]
            for i, galaxy_path, galaxy in self.path_galaxy_tuples_with_index
        }

    @property
    def image_galaxy_1d_dict(self) -> {str: g.Galaxy}:
        """
        A dictionary associating galaxy names with model images of those galaxies
        """

        image_1d_dict = {}

        for galaxy, galaxy_image_2d in self.image_2d_dict.items():

            image_1d_dict[galaxy] = self.mask_2d.array_1d_from_array_2d(
                array_2d=galaxy_image_2d
            )

        return image_1d_dict

    @property
    def hyper_galaxy_image_1d_path_dict(self):
        """
        A dictionary associating 1D hyper_galaxies galaxy images with their names.
        """

        hyper_minimum_percent = af.conf.instance.general.get(
            "hyper", "hyper_minimum_percent", float
        )

        hyper_galaxy_image_1d_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:

            galaxy_image_1d = self.image_galaxy_1d_dict[path]

            minimum_galaxy_value = hyper_minimum_percent * max(galaxy_image_1d)
            galaxy_image_1d[
                galaxy_image_1d < minimum_galaxy_value
            ] = minimum_galaxy_value
            hyper_galaxy_image_1d_path_dict[path] = galaxy_image_1d

        return hyper_galaxy_image_1d_path_dict

    @property
    def hyper_galaxy_image_2d_path_dict(self):
        """
        A dictionary associating 2D hyper_galaxies galaxy images with their names.
        """

        hyper_galaxy_image_2d_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:

            hyper_galaxy_image_2d_path_dict[
                path
            ] = self.mask_2d.scaled_array_2d_from_array_1d(
                array_1d=self.hyper_galaxy_image_1d_path_dict[path]
            )

        return hyper_galaxy_image_2d_path_dict

    def binned_image_1d_dict_from_binned_grid(self, binned_grid) -> {str: g.Galaxy}:
        """
        A dictionary associating 1D cluster images with their names.
        """

        binned_image_1d_dict = {}

        for galaxy, galaxy_image_2d in self.image_2d_dict.items():

            binned_image_2d = binning_util.binned_up_array_2d_using_mean_from_array_2d_and_bin_up_factor(
                array_2d=galaxy_image_2d, bin_up_factor=binned_grid.bin_up_factor
            )

            binned_image_1d_dict[galaxy] = binned_grid.mask.array_1d_from_array_2d(
                array_2d=binned_image_2d
            )

        return binned_image_1d_dict

    def binned_hyper_galaxy_image_1d_path_dict_from_binned_grid(self, binned_grid):
        """
        A dictionary associating 1D hyper_galaxies galaxy cluster images with their names.
        """

        if binned_grid is not None:

            hyper_minimum_percent = af.conf.instance.general.get(
                "hyper", "hyper_minimum_percent", float
            )

            binned_image_1d_galaxy_dict = self.binned_image_1d_dict_from_binned_grid(
                binned_grid=binned_grid
            )

            binned_hyper_galaxy_image_path_dict = {}

            for path, galaxy in self.path_galaxy_tuples:
                binned_galaxy_image_1d = binned_image_1d_galaxy_dict[path]

                minimum_hyper_value = hyper_minimum_percent * max(
                    binned_galaxy_image_1d
                )
                binned_galaxy_image_1d[
                    binned_galaxy_image_1d < minimum_hyper_value
                ] = minimum_hyper_value

                binned_hyper_galaxy_image_path_dict[path] = binned_galaxy_image_1d

            return binned_hyper_galaxy_image_path_dict

    def binned_hyper_galaxy_image_2d_path_dict_from_binned_grid(self, binned_grid):
        """
        A dictionary associating "D hyper_galaxies galaxy images cluster images with their names.
        """

        if binned_grid is not None:

            binned_hyper_galaxy_image_1d_path_dict = self.binned_hyper_galaxy_image_1d_path_dict_from_binned_grid(
                binned_grid=binned_grid
            )

            binned_hyper_galaxy_image_2d_path_dict = {}

            for path, galaxy in self.path_galaxy_tuples:
                binned_hyper_galaxy_image_2d_path_dict[
                    path
                ] = binned_grid.mask.scaled_array_2d_from_array_1d(
                    array_1d=binned_hyper_galaxy_image_1d_path_dict[path]
                )

            return binned_hyper_galaxy_image_2d_path_dict

    @property
    def hyper_model_image_1d(self):

        hyper_model_image_1d = np.zeros(self.mask_2d.pixels_in_mask)

        for path, galaxy in self.path_galaxy_tuples:
            hyper_model_image_1d += self.hyper_galaxy_image_1d_path_dict[path]

        return hyper_model_image_1d


class MockResult:
    def __init__(self, constant, figure_of_merit, variable=None):
        self.constant = constant
        self.figure_of_merit = figure_of_merit
        self.variable = variable
        self.previous_variable = variable
        self.gaussian_tuples = None
        self.mask_2d = None
        self.positions = None


class MockHyperCombinedPhase(object):
    def __init__(self):

        pass

    @property
    def most_likely_pixelization_grids_of_planes(self):
        return 1


class MockNLO(af.NonLinearOptimizer):
    def fit(self, analysis):
        class Fitness(object):
            def __init__(self, instance_from_physical_vector):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector

            def __call__(self, vector):
                instance = self.instance_from_physical_vector(vector)

                likelihood = analysis.fit(instance)
                self.result = MockResult(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector)
        fitness_function(self.variable.prior_count * [0.8])

        return fitness_function.result
