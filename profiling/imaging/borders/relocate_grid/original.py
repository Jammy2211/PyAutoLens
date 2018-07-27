import numpy as np

from src.profiles import light_profiles
from profiling import profiling_data
from profiling import tools
from imaging import mask

class SubGridBorderProfiling(mask.SubGridBorder):

    def __new__(cls, arr, polynomial_degree=3, centre=(0.0, 0.0), *args, **kwargs):
        border = arr.view(cls)
        border.polynomial_degree = polynomial_degree
        border.centre = centre
        return border

    @classmethod
    def from_mask(cls, mask, sub_grid_size, polynomial_degree=3, centre=(0.0, 0.0)):
        return cls(mask.border_sub_pixel_indices(sub_grid_size), polynomial_degree, centre)

    def relocated_grid_from_grid(self, grid):
        move_factors = self.move_factors_from_grid(grid)
        return np.multiply(grid, move_factors[:, None])
    
sub_grid_size = 4
    
lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size)
lsst_border = SubGridBorderProfiling.from_mask(mask=lsst.mask, sub_grid_size=sub_grid_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size)
euclid_border = SubGridBorderProfiling.from_mask(mask=euclid.mask, sub_grid_size=sub_grid_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size)
hst_border = SubGridBorderProfiling.from_mask(mask=hst.mask, sub_grid_size=sub_grid_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size)
hst_up_border = SubGridBorderProfiling.from_mask(mask=hst_up.mask, sub_grid_size=sub_grid_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=sub_grid_size)
ao_border = SubGridBorderProfiling.from_mask(mask=ao.mask, sub_grid_size=sub_grid_size)

@tools.tick_toc_x1
def lsst_solution():
    lsst_border.relocated_grid_from_grid(grid=lsst.grids.sub)

@tools.tick_toc_x1
def euclid_solution():
    euclid_border.relocated_grid_from_grid(grid=euclid.grids.sub)

@tools.tick_toc_x1
def hst_solution():
    hst_border.relocated_grid_from_grid(grid=hst.grids.sub)

@tools.tick_toc_x1
def hst_up_solution():
    hst_up_border.relocated_grid_from_grid(grid=hst_up.grids.sub)

@tools.tick_toc_x1
def ao_solution():
    ao_border.relocated_grid_from_grid(grid=ao.grids.sub)
    
if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()