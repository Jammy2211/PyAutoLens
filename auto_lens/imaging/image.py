import numpy as np


class DataGrid(np.ndarray):
    def __new__(cls, array, pixel_scale=1):
        return np.array(array).view(cls)

    # noinspection PyUnusedLocal
    def __init__(self, array, pixel_scale):
        # noinspection PyArgumentList
        super(DataGrid, self).__init__()
        self.pixel_scale = pixel_scale

    @property
    def central_pixel_coordinates(self):
        return float(self.shape[0] - 1) / 2, float(self.shape[1] - 1) / 2

    def pixels_to_arc_seconds(self, pixels):
        return self.pixel_scale * pixels

    def arc_seconds_to_pixels(self, arc_seconds):
        return arc_seconds / self.pixel_scale

    def pixel_coordinates_to_arc_second_coordinates(self, pixel_coordinates):
        return tuple(map(lambda coord, centre: self.pixels_to_arc_seconds(coord - centre), pixel_coordinates,
                         self.central_pixel_coordinates))

    def arc_second_coordinates_to_pixel_coordinates(self, arc_second_coordinates):
        return tuple(map(lambda coord, centre: self.arc_seconds_to_pixels(coord) + centre,
                         arc_second_coordinates,
                         self.central_pixel_coordinates))
