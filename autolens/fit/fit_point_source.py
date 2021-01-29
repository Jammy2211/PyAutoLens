from autoarray.structures import arrays, grids
from autoarray.util import fit_util
from autoarray.fit.fit import FitData
import numpy as np


class AbstractFitPositionsSourcePlane:
    def __init__(self, positions, noise_map, tracer):
        """Given a positions dataset, which is a list of positions with names that associated them to model source
        galaxies, use a `Tracer` to determine the traced coordinate positions in the source-plane.

        Different children of this abstract class are available which use the traced coordinates to define a chi-squared
        value in different ways.

        Parameters
        -----------
        positions : grids.Grid2DIrregularGrouped
            The (y,x) arc-second coordinates of named positions which the log_likelihood is computed using. Positions
            are paired to galaxies in the `Tracer` using their names.
        tracer : ray_tracing.Tracer
            The object that defines the ray-tracing of the strong lens system of galaxies.
        noise_value : float
            The noise-value assumed when computing the log likelihood.
        """
        self.positions = positions
        self.noise_map = noise_map
        self.source_plane_positions = tracer.traced_grids_of_planes_from_grid(
            grid=positions
        )[-1]

    @property
    def furthest_separations_of_source_plane_positions(
        self
    ) -> arrays.ValuesIrregularGrouped:
        """
        Returns the furthest distance of every source-plane (y,x) coordinate to the other source-plane (y,x)
        coordinates. This is performed separatly for each grouped set of source-plane coordinates.

        For example, for the following source-plane positions:

        source_plane_positions = [[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)], [(0.0, 0.0), (1.0, 1.0)]]

        The returned furthest distances are:

        source_plane_positions = [[3.0, 2.0, 3.0], [1.0, 1.0]

        Returns
        -------
        arrays.ValuesIrregularGrouped
            The further distances of every set of grouped source-plane coordinates the other source-plane coordinates
            that it is grouped with.
        """
        return self.source_plane_positions.furthest_distances_from_other_coordinates

    @property
    def max_separation_of_source_plane_positions(self) -> float:
        return max(self.furthest_separations_of_source_plane_positions)

    def max_separation_within_threshold(self, threshold) -> bool:
        return self.max_separation_of_source_plane_positions <= threshold


class FitPositionsSourceMaxSeparation(AbstractFitPositionsSourcePlane):
    def __init__(self, positions, noise_map, tracer):
        """A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher log_likelihood.

        Parameters
        -----------
        positions : grids.Grid2DIrregularGrouped
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value : float
            The noise-value assumed when computing the log likelihood.
        """
        super().__init__(positions=positions, noise_map=noise_map, tracer=tracer)

    # @property
    # def chi_squared_map(self):
    #     return np.square(np.divide(self.max_separation_of_source_plane_positions, self.noise_map))
    #
    # @property
    # def figure_of_merit(self):
    #     return -0.5 * sum(self.chi_squared_map)


class FitPositionsImage(FitData):
    def __init__(self, positions, noise_map, tracer, positions_solver):
        """A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher log_likelihood.

        Parameters
        -----------
        positions : grids.Grid2DIrregularGrouped
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value : float
            The noise-value assumed when computing the log likelihood.
        """

        self.positions_solver = positions_solver
        self.model_positions_all = positions_solver.solve_from_tracer(tracer=tracer)

        model_positions = self.model_positions_all.grid_of_closest_from_grid_pair(
            grid_pair=positions
        )

        super().__init__(
            data=positions,
            noise_map=noise_map,
            model_data=model_positions,
            mask=None,
            inversion=None,
        )

    @property
    def positions(self):
        return self.data

    @property
    def model_positions(self):
        return self.model_data

    @property
    def residual_map(self) -> grids.Grid2DIrregularGrouped:
        residual_positions = self.positions - self.model_positions
        return residual_positions.distances_from_coordinate(coordinate=(0.0, 0.0))


class FitFluxes(FitData):
    def __init__(self, fluxes, noise_map, positions, tracer):

        # TODO : The fluxes, positions etc that come into here will be IrregularGrouped structures with dictionary inputs.
        # TODO : We need them as numpy array sso that we caninherit and subtract efficiently. Easy to do.
        # TODO : These can be generated from PointSourceData classes.

        self.positions = positions
        self.magnifications = abs(
            tracer.magnification_via_hessian_from_grid(grid=positions)
        )

        model_fluxes = arrays.ValuesIrregularGrouped(
            values=[
                magnification * tracer.flux_hack
                for magnification in self.magnifications
            ]
        )

        super().__init__(
            data=fluxes,
            noise_map=noise_map,
            model_data=model_fluxes,
            mask=None,
            inversion=None,
        )

    @property
    def fluxes(self):
        return self.data

    @property
    def model_fluxes(self):
        return self.model_data
