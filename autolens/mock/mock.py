import autofit as af

from autofit.mock.mock import MockSearch, MockSamples
from autoarray.mock.mock import MockMask, MockDataset, MockFit as AAMockFit
from autogalaxy.mock.mock import MockLightProfile, MockMassProfile

from autofit.mock.mock import *
from autofit.mock import mock as af_m


class MockResult(af_m.MockResult):
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
        max_log_likelihood_pixelization_grids_of_planes=None,
        hyper_galaxy_image_path_dict=None,
        hyper_model_image=None,
        hyper_galaxy_visibilities_path_dict=None,
        hyper_model_visibilities=None,
        pixelization=None,
        positions=None,
        updated_positions=None,
        updated_positions_threshold=None,
        stochastic_log_likelihoods=None,
    ):

        super().__init__(
            samples=samples,
            instance=instance,
            model=model,
            analysis=analysis,
            search=search,
        )

        self._model = model
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
        self.max_log_likelihood_pixelization_grids_of_planes = (
            max_log_likelihood_pixelization_grids_of_planes
        )
        self.pixelization = pixelization
        self.positions = positions
        self.updated_positions = (
            updated_positions if updated_positions is not None else []
        )
        self.updated_positions_threshold = updated_positions_threshold
        self._stochastic_log_likelihoods = stochastic_log_likelihoods

    def stochastic_log_likelihoods(self):
        return self._stochastic_log_likelihoods

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
        stochastic_log_likelihoods=None,
    ):
        """
        A collection of results from previous searchs. Results can be obtained using an index or the name of the search
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
            stochastic_log_likelihoods=stochastic_log_likelihoods,
        )

        self.__result_list = [result]

    @property
    def last(self):
        """
        The result of the last search
        """
        if len(self.__result_list) > 0:
            return self.__result_list[-1]
        return None

    def __getitem__(self, item):
        """
        Get the result of a previous search by index

        Parameters
        ----------
        item: int
            The index of the result

        Returns
        -------
        result: Result
            The result of a previous search
        """
        return self.__result_list[item]

    def __len__(self):
        return len(self.__result_list)


class MockFit(AAMockFit):
    def __init__(
        self,
        tracer=None,
        dataset=MockDataset(),
        inversion=None,
        noise_map=None,
        grid=None,
        blurred_image=None,
    ):

        super().__init__(dataset=dataset, inversion=inversion, noise_map=noise_map)

        self.tracer = tracer
        self.grid = grid
        self.blurred_image = blurred_image


class MockTracer:
    def __init__(
        self, traced_grids_of_planes=None, sparse_image_plane_grid_pg_list=None
    ):

        self.traced_grids_of_planes = traced_grids_of_planes
        self.sparse_image_plane_grid_pg_list = sparse_image_plane_grid_pg_list

    def traced_grid_list_from(self, grid):

        return self.traced_grids_of_planes

    def sparse_image_plane_grid_pg_list_from(self, grid):

        return self.sparse_image_plane_grid_pg_list


class MockTracerPoint(MockTracer):
    def __init__(
        self,
        sparse_image_plane_grid_pg_list=None,
        traced_grid=None,
        attribute=None,
        profile=None,
        magnification=None,
        einstein_radius=None,
        einstein_mass=None,
    ):

        super().__init__(
            sparse_image_plane_grid_pg_list=sparse_image_plane_grid_pg_list
        )

        self.positions = traced_grid

        self.attribute = attribute
        self.profile = profile

        self.magnification = magnification
        self.einstein_radius = einstein_radius
        self.einstein_mass = einstein_mass

    @property
    def planes(self):
        return [0, 1]

    def deflections_yx_2d_from(self):
        pass

    @property
    def has_mass_profile(self):
        return True

    def extract_attribute(self, cls, attr_name):
        return [self.attribute]

    def extract_profile(self, profile_name):
        return self.profile

    def traced_grid_list_from(self, grid, plane_index_limit=None):
        return [self.positions]

    def magnification_2d_via_hessian_from(self, grid, deflections_func=None):
        return self.magnification

    def einstein_radius_from(self, grid):
        return self.einstein_radius

    def einstein_mass_angular_from(self, grid):
        return self.einstein_mass


class MockPointSolver:
    def __init__(self, model_positions):

        self.model_positions = model_positions

    def solve(self, lensing_obj, source_plane_coordinate, upper_plane_index=None):
        return self.model_positions
