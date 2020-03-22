import numpy as np

import autofit as af
import autolens as al


class GalaxiesMockAnalysis:
    def __init__(self, number_galaxies, value):
        self.number_galaxies = number_galaxies
        self.value = value

    # noinspection PyUnusedLocal
    def galaxy_images_for_model(self, model):
        return self.number_galaxies * [np.array([self.value])]

    def fit(self, instance):
        return 1


class MockResults:
    def __init__(
        self,
        mask=None,
        model_image=None,
        galaxy_images=(),
        model_visibilities=None,
        galaxy_visibilities=(),
        instance=None,
        analysis=None,
        optimizer=None,
        pixelization=None,
    ):
        self.mask_2d = mask
        self.model_image = model_image
        self.unmasked_model_image = model_image
        self.galaxy_images = galaxy_images
        self.model_visibilities = model_visibilities
        self.galaxy_visibilities = galaxy_visibilities
        self.instance = instance or af.ModelInstance()
        self.model = af.ModelMapper()
        self.analysis = analysis
        self.optimizer = optimizer
        self.pixelization = pixelization
        self.hyper_combined = MockHyperCombinedPhase()
        self.use_as_hyper_dataset = False

    @property
    def path_galaxy_tuples(self) -> [(str, al.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return [(("g0",), al.Galaxy(redshift=0.5)), (("g1",), al.Galaxy(redshift=1.0))]

    @property
    def path_galaxy_tuples_with_index(self) -> [(str, al.Galaxy)]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return [
            (0, ("g0",), al.Galaxy(redshift=0.5)),
            (1, ("g1",), al.Galaxy(redshift=1.0)),
        ]

    @property
    def image_galaxy_dict(self) -> {str: al.Galaxy}:
        """
        A dictionary associating galaxy names with model images of those galaxies
        """
        return {
            galaxy_path: self.galaxy_images[i]
            for i, galaxy_path, galaxy in self.path_galaxy_tuples_with_index
        }

    @property
    def hyper_galaxy_image_path_dict(self):
        """
        A dictionary associating 1D hyper_galaxies galaxy images with their names.
        """

        hyper_minimum_percent = af.conf.instance.general.get(
            "hyper", "hyper_minimum_percent", float
        )

        hyper_galaxy_image_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:
            galaxy_image = self.image_galaxy_dict[path]

            minimum_galaxy_value = hyper_minimum_percent * max(galaxy_image)

            galaxy_image[galaxy_image < minimum_galaxy_value] = minimum_galaxy_value

            hyper_galaxy_image_path_dict[path] = galaxy_image

        return hyper_galaxy_image_path_dict

    @property
    def hyper_model_image(self):

        hyper_model_image = al.masked_array.zeros(mask=self.mask_2d)

        for path, galaxy in self.path_galaxy_tuples:
            hyper_model_image += self.hyper_galaxy_image_path_dict[path]

        return hyper_model_image

    @property
    def visibilities_galaxy_dict(self) -> {str: al.Galaxy}:
        """
        A dictionary associating galaxy names with model visibilities of those galaxies
        """
        return {
            galaxy_path: self.galaxy_visibilities[i]
            for i, galaxy_path, galaxy in self.path_galaxy_tuples_with_index
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

        hyper_model_visibilities = al.visibilities.zeros(
            shape_1d=(self.galaxy_visibilities[0].shape_1d,)
        )

        for path, galaxy in self.path_galaxy_tuples:
            hyper_model_visibilities += self.hyper_galaxy_visibilities_path_dict[path]

        return hyper_model_visibilities


class MockResult:
    def __init__(self, instance, likelihood, model=None):
        self.instance = instance
        self.likelihood = likelihood
        self.model = model
        self.previous_model = model
        self.gaussian_tuples = None
        self.mask_2d = None
        self.positions = None


class MockHyperCombinedPhase:
    def __init__(self):
        pass

    @property
    def most_likely_pixelization_grids_of_planes(self):
        return 1


class MockNLO(af.NonLinearOptimizer):
    def fit(self, analysis, model):
        class Fitness:
            def __init__(self, instance_from_vector):
                self.result = None
                self.instance_from_vector = instance_from_vector

            def __call__(self, vector):
                instance = self.instance_from_vector(vector)

                likelihood = analysis.fit(instance)
                self.result = MockResult(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(model.instance_from_vector)
        fitness_function(model.prior_count * [0.8])

        return fitness_function.result
