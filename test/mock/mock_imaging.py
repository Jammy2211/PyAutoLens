import numpy as np

from autolens.imaging import mask as msk


class MockSubGrid(np.ndarray):

    def __new__(cls, sub_grid, *args, **kwargs):
        return sub_grid.view(cls)

    def __init__(self, sub_grid, sub_to_image, sub_grid_size):
        # noinspection PyArgumentList
        super().__init__()
        self.sub_grid_coords = sub_grid
        self.sub_to_image = sub_to_image
        self.total_pixels = sub_to_image.shape[0]
        self.sub_grid_size = sub_grid_size
        self.sub_grid_length = int(sub_grid_size ** 2.0)
        self.sub_grid_fraction = 1.0 / self.sub_grid_length


class MockGridCollection(object):

    def __init__(self, image, sub, blurring=None):
        self.image = msk.ImageGrid(image, mask=None)
        self.sub = sub
        self.blurring = msk.ImageGrid(blurring, mask=None) if blurring is not None else None


class MockBorders(object):

    def __init__(self, image=None, sub=None):
        self.image = image
        self.sub = sub