import numpy as np
from autolens.data.array import grids

@grids.sub_to_image_grid
def intensities_of_galaxies_from_grid(grid, galaxies):
    return sum(map(lambda g: g.intensities_from_grid(grid), galaxies))

@grids.sub_to_image_grid
def surface_density_of_galaxies_from_grid(grid, galaxies):
    return sum(map(lambda g: g.surface_density_from_grid(grid), galaxies))

@grids.sub_to_image_grid
def potential_of_galaxies_from_grid(grid, galaxies):
    return sum(map(lambda g: g.potential_from_grid(grid), galaxies))

def deflections_of_galaxies_from_grid(grid, galaxies):
    deflections = sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))
    if isinstance(grid, grids.SubGrid):
        return np.asarray([grid.sub_data_to_regular_data(deflections[:, 0]),
                           grid.sub_data_to_regular_data(deflections[:, 1])]).T
    return sum(map(lambda galaxy: galaxy.deflections_from_grid(grid), galaxies))

def deflections_of_galaxies_from_sub_grid(sub_grid, galaxies):
    return sum(map(lambda galaxy: galaxy.deflections_from_grid(sub_grid), galaxies))

def deflections_of_galaxies_from_grid_stack(grid_stack, galaxies):
    return grid_stack.apply_function(lambda grid: deflections_of_galaxies_from_sub_grid(grid, galaxies))