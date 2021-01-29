import autofit as af
from autofit.mock.mock import MockSearch, MockSamples


class MockResult(af.MockResult):
    def __init__(
        self,
        samples=None,
        instance=None,
        model=None,
        analysis=None,
        search=None,
        mask=None,
        model_image=None,
        max_log_likelihood_tracer=None,
        max_log_likelihood_fit=None,
        hyper_galaxy_image_path_dict=None,
        hyper_model_image=None,
        hyper_galaxy_visibilities_path_dict=None,
        hyper_model_visibilities=None,
        pixelization=None,
        positions=None,
        updated_positions=None,
        updated_positions_threshold=None,
        stochastic_log_evidences=None,
        use_as_hyper_dataset=False,
    ):
        super().__init__(
            samples=samples,
            instance=instance,
            model=model,
            analysis=analysis,
            search=search,
        )

        self.previous_model = model
        self.gaussian_tuples = None
        self.mask = None
        self.positions = None
        self.mask = mask
        self.hyper_galaxy_image_path_dict = hyper_galaxy_image_path_dict
        self.hyper_model_image = hyper_model_image
        self.hyper_galaxy_visibilities_path_dict = hyper_galaxy_visibilities_path_dict
        self.hyper_model_visibilities = hyper_model_visibilities
        self.model_image = model_image
        self.unmasked_model_image = model_image
        self.max_log_likelihood_tracer = max_log_likelihood_tracer
        self.max_log_likelihood_fit = max_log_likelihood_fit
        self.pixelization = pixelization
        self.use_as_hyper_dataset = use_as_hyper_dataset
        self.positions = positions
        self.updated_positions = (
            updated_positions if updated_positions is not None else []
        )
        self.updated_positions_threshold = updated_positions_threshold
        self._stochastic_log_evidences = stochastic_log_evidences

    def stochastic_log_evidences(self):
        return self._stochastic_log_evidences

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
        search=None,
        mask=None,
        model_image=None,
        max_log_likelihood_tracer=None,
        hyper_galaxy_image_path_dict=None,
        hyper_model_image=None,
        hyper_galaxy_visibilities_path_dict=None,
        hyper_model_visibilities=None,
        pixelization=None,
        positions=None,
        updated_positions=None,
        updated_positions_threshold=None,
        stochastic_log_evidences=None,
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
            search=search,
            mask=mask,
            model_image=model_image,
            max_log_likelihood_tracer=max_log_likelihood_tracer,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
            hyper_galaxy_visibilities_path_dict=hyper_galaxy_visibilities_path_dict,
            hyper_model_visibilities=hyper_model_visibilities,
            pixelization=pixelization,
            positions=positions,
            updated_positions=updated_positions,
            updated_positions_threshold=updated_positions_threshold,
            stochastic_log_evidences=stochastic_log_evidences,
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


class MockFit:
    def __init__(self, grid):

        self.grid = grid


class MockTracer:
    def __init__(
        self,
        traced_grid=None,
        einstein_radius=None,
        einstein_mass=None,
        magnification=None,
        flux_hack=None,
    ):

        self.positions = traced_grid
        self.magnification = magnification
        self.flux_hack = flux_hack

        self.einstein_radius = einstein_radius
        self.einstein_mass = einstein_mass

    @property
    def has_mass_profile(self):
        return True

    def traced_grids_of_planes_from_grid(self, grid, plane_index_limit=None):
        return [self.positions]

    def magnification_via_hessian_from_grid(self, grid):
        return self.magnification

    def einstein_radius_from_grid(self, grid):
        return self.einstein_radius

    def einstein_mass_angular_from_grid(self, grid):
        return self.einstein_mass


class MockPositionsSolver:
    def __init__(self, model_positions):

        self.model_positions = model_positions

    def solve_from_tracer(self, tracer):
        return self.model_positions
