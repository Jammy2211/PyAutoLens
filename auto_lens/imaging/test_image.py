from auto_lens.imaging import image
import numpy as np


class TestDataGrid(object):
    def test__constructor(self):
        data_grid = image.DataGrid(np.zeros((2, 2)), pixel_scale=1)
        assert data_grid.shape == (2, 2)
        assert data_grid.pixel_scale == 1
        assert isinstance(data_grid, np.ndarray)
        assert isinstance(data_grid, image.DataGrid)
