import numpy as np

from autoarray.fit import fit as aa_fit
from autogalaxy.galaxy import galaxy as g


class AbstractFitPositionsSourcePlane:
    def __init__(self, positions, tracer, noise_value):
        """Given a positions dataset, which is a list of positions with names that associated them to model source
        galaxies, use a `Tracer` to determine the traced coordinate positions in the source-plane.

        Different children of this abstract class are available which use the traced coordinates to define a chi-squared
        value in different ways.

        Parameters
        -----------
        positions : grids.GridCoordinates
            The (y,x) arc-second coordinates of named positions which the log_likelihood is computed using. Positions
            are paired to galaxies in the `Tracer` using their names.
        tracer : ray_tracing.Tracer
            The object that defines the ray-tracing of the strong lens system of galaxies.
        noise_value : float
            The noise-value assumed when computing the log likelihood.
        """
        self.positions = positions
        self.source_plane_positions = tracer.traced_grids_of_planes_from_grid(
            grid=positions
        )[-1]
        self.noise_value = noise_value

    @property
    def maximum_separations(self):
        return [
            self.max_separation_of_grid(grid=np.asarray(positions))
            for positions in self.source_plane_positions.in_list
        ]

    def maximum_separation_within_threshold(self, threshold):
        return max(self.maximum_separations) <= threshold

    @staticmethod
    def max_separation_of_grid(grid):
        rdist_max = np.zeros((grid.shape[0]))
        for i in range(grid.shape[0]):
            xdists = np.square(np.subtract(grid[i, 0], grid[:, 0]))
            ydists = np.square(np.subtract(grid[i, 1], grid[:, 1]))
            rdist_max[i] = np.max(np.add(xdists, ydists))
        return np.max(np.sqrt(rdist_max))


class FitPositionsSourcePlaneMaxSeparation(AbstractFitPositionsSourcePlane):
    def __init__(self, positions, tracer, noise_value):
        """A lens position fitter, which takes a set of positions (e.g. from a plane in the tracer) and computes \
        their maximum separation, such that points which tracer closer to one another have a higher log_likelihood.

        Parameters
        -----------
        positions : grids.GridCoordinates
            The (y,x) arc-second coordinates of positions which the maximum distance and log_likelihood is computed using.
        noise_value : float
            The noise-value assumed when computing the log likelihood.
        """
        super(FitPositionsSourcePlaneMaxSeparation, self).__init__(
            positions=positions, tracer=tracer, noise_value=noise_value
        )

    @property
    def chi_squared_map(self):
        return np.square(np.divide(self.maximum_separations, self.noise_value))

    @property
    def figure_of_merit(self):
        return -0.5 * sum(self.chi_squared_map)
