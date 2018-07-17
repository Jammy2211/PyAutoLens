import numpy as np

from src.imaging import mask


class MockSubGridCoords(np.ndarray):
    def __new__(cls, sub_grid_coords, *args, **kwargs):
        return sub_grid_coords.view(cls)

    def __init__(self, sub_grid_coords, sub_to_image, sub_grid_size):
        # noinspection PyArgumentList
        super().__init__()
        self.sub_grid_coords = sub_grid_coords
        self.sub_to_image = sub_to_image
        self.no_pixels = sub_to_image.shape[0]
        self.sub_grid_size = sub_grid_size
        self.sub_grid_length = int(sub_grid_size ** 2.0)
        self.sub_grid_fraction = 1.0 / self.sub_grid_length


class MockCoordinateCollection(object):

    def __init__(self, image_coords, sub_grid_coords, blurring_grid=None):
        self.image_coords = mask.CoordinateGrid(image_coords)
        self.sub_grid_coords = sub_grid_coords
        self.blurring_grid = mask.CoordinateGrid(blurring_grid) if blurring_grid is not None else None
