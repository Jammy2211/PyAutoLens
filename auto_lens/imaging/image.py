import numpy as np


class DataGrid(np.ndarray):
    def __new__(cls, array, pixel_scale):
        return np.array(array).view(cls)

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale):
        # noinspection PyArgumentList
        super(DataGrid, self).__init__()
        self.pixel_scale = pixel_scale
