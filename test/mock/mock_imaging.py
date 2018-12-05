import numpy as np

from autolens.data.array import grids


class MockSubGrid(np.ndarray):

    def __new__(cls, sub_grid, *args, **kwargs):
        return sub_grid.view(cls)

    def __init__(self, sub_grid, sub_to_regular, sub_grid_size):
        # noinspection PyArgumentList
        super().__init__()
        self.sub_grid_coords = sub_grid
        self.sub_to_regular = sub_to_regular
        self.total_pixels = sub_to_regular.shape[0]
        self.sub_grid_size = sub_grid_size
        self.sub_grid_length = int(sub_grid_size ** 2.0)
        self.sub_grid_fraction = 1.0 / self.sub_grid_length


class MockGridCollection(object):

    def __init__(self, regular, sub, blurring=None, pix=None, regular_to_nearest_regular_pix=None):
        self.regular = grids.RegularGrid(regular, mask=None)
        self.sub = sub
        self.blurring = grids.RegularGrid(blurring, mask=None) if blurring is not None else None
        self.pix = grids.PixGrid(pix, regular_to_nearest_regular_pix=regular_to_nearest_regular_pix,
                                 mask=None) if pix is not None else np.array([[0.0, 0.0]])



class MockBorders(object):

    def __init__(self, regular=None, sub=None):
        self.regular = regular
        self.sub = sub