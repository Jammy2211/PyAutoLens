import numpy as np

import autofit as af
import autolens as al


class GalaxiesMockAnalysis:
    def __init__(self, number_galaxies, value):
        self.number_galaxies = number_galaxies
        self.value = value
        self.hyper_model_image = None
        self.hyper_galaxy_image_path_dict = None

    # noinspection PyUnusedLocal
    def galaxy_images_for_model(self, model):
        return self.number_galaxies * [np.array([self.value])]

    def log_likelihood_function(self, instance):
        return 1


class MockSamples(af.AbstractSamples):
    def __init__(
        self,
        max_log_likelihood_instance=None,
        log_likelihoods=None,
        gaussian_tuples=None,
    ):

        if log_likelihoods is None:
            log_likelihoods = [1.0, 2.0, 3.0]

        super().__init__(
            model=None,
            parameters=[],
            log_likelihoods=log_likelihoods,
            log_priors=[],
            weights=[],
        )

        self._max_log_likelihood_instance = max_log_likelihood_instance
        self.gaussian_tuples = gaussian_tuples

    @property
    def max_log_likelihood_instance(self) -> int:
        return self._max_log_likelihood_instance

    def gaussian_priors_at_sigma(self, sigma=None):
        return self.gaussian_tuples

    @property
    def max_log_likelihood_instance(self):

        if self._max_log_likelihood_instance is not None:
            return self._max_log_likelihood_instance

        self.galaxies = [
            al.Galaxy(redshift=0.5, light=al.lp.EllipticalSersic(centre=(0.0, 1.0))),
            al.Galaxy(redshift=1.0, light=al.lp.EllipticalSersic()),
        ]

        return self


class MockResult:
    def __init__(
        self,
        samples=None,
        instance=None,
        model=None,
        analysis=None,
        optimizer=None,
        mask=None,
        model_image=None,
        hyper_galaxy_image_path_dict=None,
        hyper_model_image=None,
        hyper_galaxy_visibilities_path_dict=None,
        hyper_model_visibilities=None,
        pixelization=None,
        positions=None,
        updated_positions=None,
        updated_positions_threshold=None,
        use_as_hyper_dataset=False,
    ):

        self.instance = instance or af.ModelInstance()
        self.model = model or af.ModelMapper()
        self.samples = samples or MockSamples(max_log_likelihood_instance=self.instance)

        self.previous_model = model
        self.gaussian_tuples = None
        self.mask_2d = None
        self.positions = None
        self.mask_2d = mask
        self.hyper_galaxy_image_path_dict = hyper_galaxy_image_path_dict
        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_visibilities_path_dict = hyper_galaxy_visibilities_path_dict
        self.hyper_model_visibilities = hyper_model_visibilities
        self.model_image = model_image
        self.unmasked_model_image = model_image
        self.analysis = analysis
        self.optimizer = optimizer
        self.pixelization = pixelization
        self.hyper_combined = MockHyperCombinedPhase()
        self.use_as_hyper_dataset = use_as_hyper_dataset
        self.positions = positions
        self.updated_positions = (
            updated_positions if updated_positions is not None else []
        )
        self.updated_positions_threshold = updated_positions_threshold
        self.max_log_likelihood_tracer = al.Tracer.from_galaxies(
            galaxies=[al.Galaxy(redshift=0.5)]
        )

    @property
    def last(self):
        return self

    @property
    def image_plane_multiple_image_positions_of_source_plane_centres(self):
        return self.updated_positions


class MockResults(af.ResultsCollection):
    def __init__(
        self,
        samples=None,
        instance=None,
        model=None,
        analysis=None,
        optimizer=None,
        mask=None,
        model_image=None,
        hyper_galaxy_image_path_dict=None,
        hyper_model_image=None,
        hyper_galaxy_visibilities_path_dict=None,
        hyper_model_visibilities=None,
        pixelization=None,
        positions=None,
        updated_positions=None,
        updated_positions_threshold=None,
        use_as_hyper_dataset=False,
    ):
        """
        A collection of results from previous phases. Results can be obtained using an index or the name of the phase
        from whence they came.
        """

        super().__init__()

        result = MockResult(
            samples=samples,
            instance=instance,
            model=model,
            analysis=analysis,
            optimizer=optimizer,
            mask=mask,
            model_image=model_image,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
            hyper_galaxy_visibilities_path_dict=hyper_galaxy_visibilities_path_dict,
            hyper_model_visibilities=hyper_model_visibilities,
            pixelization=pixelization,
            positions=positions,
            updated_positions=updated_positions,
            updated_positions_threshold=updated_positions_threshold,
            use_as_hyper_dataset=use_as_hyper_dataset,
        )

        self.__result_list = [result]

    @property
    def last(self):
        """
        The result of the last phase
        """
        if len(self.__result_list) > 0:
            return self.__result_list[-1]
        return None

    def __getitem__(self, item):
        """
        Get the result of a previous phase by index

        Parameters
        ----------
        item: int
            The index of the result

        Returns
        -------
        result: Result
            The result of a previous phase
        """
        return self.__result_list[item]

    def __len__(self):
        return len(self.__result_list)


class MockHyperCombinedPhase:
    def __init__(self):
        pass

    @property
    def max_log_likelihood_pixelization_grids_of_planes(self):
        return 1


class MockNLO(af.NonLinearOptimizer):
    def _fit(self, analysis, fitness_function):
        # noinspection PyTypeChecker
        return af.Result(None, analysis.log_likelihood_function(None), None)

    def _full_fit(self, model, analysis):
        class Fitness:
            def __init__(self, instance_from_vector):
                self.result = None
                self.instance_from_vector = instance_from_vector

            def __call__(self, vector):
                instance = self.instance_from_vector(vector)

                log_likelihood = analysis.log_likelihood_function(instance)
                self.result = MockResult(instance=instance)

                # Return Chi squared
                return -2 * log_likelihood

        fitness_function = Fitness(model.instance_from_vector)
        fitness_function(model.prior_count * [0.8])

        return fitness_function.result

    def samples_from_model(self, model):
        return MockSamples()
