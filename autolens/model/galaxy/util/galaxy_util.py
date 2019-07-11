import numpy as np

from autolens.data.array import grids


@grids.sub_to_image_grid
def intensities_of_galaxies_from_grid(grid, galaxies):
    """Compute the intensities of a list of galaxies from an input grid, by summing the individual intensities \
    of each galaxy's light profile.

    If the input grid is a *grids.SubGrid*, the intensites is calculated on the sub-grid and binned-up to the \
    original regular grid by taking the mean value of every set of sub-pixels.

    If no galaxies are entered into the function, an array of all zeros is returned.

    Parameters
    -----------
    grid : RegularGrid
        The grid (regular or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
        intensities are calculated on.
    galaxies : [galaxy.Galaxy]
        The galaxies whose light profiles are used to compute the surface densities.
    """
    if galaxies:
        return sum(map(lambda g: g.intensities_from_grid(grid), galaxies))
    else:
        return np.full((grid.shape[0]), 0.0)


@grids.sub_to_image_grid
def convergence_of_galaxies_from_grid(grid, galaxies):
    """Compute the convergence of a list of galaxies from an input grid, by summing the individual convergence \
    of each galaxy's mass profile.

    If the input grid is a *grids.SubGrid*, the convergence is calculated on the sub-grid and binned-up to the \
    original regular grid by taking the mean value of every set of sub-pixels.

    If no galaxies are entered into the function, an array of all zeros is returned.

    Parameters
    -----------
    grid : RegularGrid
        The grid (regular or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
        convergence is calculated on.
    galaxies : [galaxy.Galaxy]
        The galaxies whose mass profiles are used to compute the convergence.
    """
    if galaxies:
        return sum(map(lambda g: g.convergence_from_grid(grid), galaxies))
    else:
        return np.full((grid.shape[0]), 0.0)


@grids.sub_to_image_grid
def potential_of_galaxies_from_grid(grid, galaxies):
    """Compute the potential of a list of galaxies from an input grid, by summing the individual potential \
    of each galaxy's mass profile.

    If the input grid is a *grids.SubGrid*, the surface-density is calculated on the sub-grid and binned-up to the \
    original regular grid by taking the mean value of every set of sub-pixels.

    If no galaxies are entered into the function, an array of all zeros is returned.

    Parameters
    -----------
    grid : RegularGrid
        The grid (regular or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
        potential is calculated on.
    galaxies : [galaxy.Galaxy]
        The galaxies whose mass profiles are used to compute the surface densities.
    """
    if galaxies:
        return sum(map(lambda g: g.potential_from_grid(grid), galaxies))
    else:
        return np.full((grid.shape[0]), 0.0)


def deflections_of_galaxies_from_grid(grid, galaxies):
    """Compute the deflections of a list of galaxies from an input grid, by summing the individual deflections \
    of each galaxy's mass profile.

    If the input grid is a *grids.SubGrid*, the potential is calculated on the sub-grid and binned-up to the \
    original regular grid by taking the mean value of every set of sub-pixels.

    If no galaxies are entered into the function, an array of all zeros is returned.

    Parameters
    -----------
    grid : RegularGrid
        The grid (regular or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
        deflections is calculated on.
    galaxies : [galaxy.Galaxy]
        The galaxies whose mass profiles are used to compute the surface densities.
    """
    if len(galaxies) > 0:
        deflections = sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))
    else:
        deflections = np.full((grid.shape[0], 2), 0.0)

    if isinstance(grid, grids.SubGrid):
        return np.asarray([grid.array_1d_binned_up_from_sub_array_1d(deflections[:, 0]),
                           grid.array_1d_binned_up_from_sub_array_1d(deflections[:, 1])]).T

    return deflections


def deflections_of_galaxies_from_sub_grid(sub_grid, galaxies):
    """Compute the deflections of a list of galaxies from an input sub-grid, by summing the individual deflections \
    of each galaxy's mass profile.

    The deflections are calculated on the sub-grid and binned-up to the original regular grid by taking the mean value \
    of every set of sub-pixels.

    If no galaxies are entered into the function, an array of all zeros is returned.

    Parameters
    -----------
    sub_grid : RegularGrid
        The grid (regular or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
        deflections is calculated on.
    galaxies : [galaxy.Galaxy]
        The galaxies whose mass profiles are used to compute the surface densities.
    """
    if galaxies:
        return sum(map(lambda galaxy: galaxy.deflections_from_grid(sub_grid), galaxies))
    else:
        return np.full((sub_grid.shape[0], 2), 0.0)


def deflections_of_galaxies_from_grid_stack(grid_stack, galaxies):
    return grid_stack.apply_function(lambda grid: deflections_of_galaxies_from_sub_grid(grid, galaxies))
