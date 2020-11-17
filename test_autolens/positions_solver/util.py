from autoarray import decorator_util
import numpy as np


def check_if_positions_in_positions_true(positions_true, positions, threshold):

    minimum_separations = minimum_separations_from(
        positions_true=positions_true, positions=positions
    )

    return [separation < threshold for separation in minimum_separations]


def minimum_separations_from(positions_true, positions):

    minimum_separations = []

    for pos_true in positions_true.in_list[0]:

        minimum_separations.append(
            min_separation_of_positions_to_grid(positions=pos_true, grid=positions)
        )

    return minimum_separations


@decorator_util.jit()
def min_separation_of_positions_to_grid(positions, grid):

    rdist_max = np.zeros((grid.shape[0]))

    for i in range(grid.shape[0]):

        xdists = np.square(np.subtract(positions[0], grid[:, 0]))
        ydists = np.square(np.subtract(positions[1], grid[:, 1]))
        rdist_max[i] = np.min(np.add(xdists, ydists))

    return np.min(np.sqrt(rdist_max))
