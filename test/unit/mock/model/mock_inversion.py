import numpy as np


class MockGeometry(object):
    def __init__(
        self,
        pixel_centres=None,
        pixel_neighbors=np.array([1]),
        pixel_neighbors_size=np.array([1]),
    ):

        self.pixel_scales = (1.0, 1.0)
        self.origin = (0.0, 0.0)
        self.pixel_centres = pixel_centres

        self.pixel_neighbors = pixel_neighbors.astype("int")
        self.pixel_neighbors_size = pixel_neighbors_size.astype("int")


class MockPixelization(object):
    def __init__(self, value, grid=None):
        self.value = value
        self.grid = grid

    # noinspection PyUnusedLocal,PyShadowingNames
    def mapper_from_grid_and_pixelization_grid(
        self, grid, pixelization_grid, inversion_uses_border, hyper_image=None
    ):
        return self.value

    def pixelization_grid_from_grid(self, grid, cluster_grid, hyper_image):
        if hyper_image is None:
            return self.grid
        else:
            return self.grid * hyper_image


class MockRegularization(object):
    def __init__(self, matrix_shape):
        self.shape = matrix_shape

    def regularization_matrix_from_pixel_neighbors(
        self, pixel_neighbors, pixel_neighbors_size
    ):
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def regularization_matrix_from_mapper(self, mapper):
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class MockMapper(object):
    def __init__(self, matrix_shape, grid=None):

        self.grid = grid
        self.mapping_matrix = np.ones(matrix_shape)
        self.geometry = MockGeometry()


class MockConvolver(object):
    def __init__(self, matrix_shape):
        self.shape = matrix_shape

    def convolve_mapping_matrix(self, mapping_matrix):
        return np.ones(self.shape)


class MockInversion(object):
    def __init__(self):
        self.blurred_mapping_matrix = np.zeros((1, 1))
        self.regularization_matrix = np.zeros((1, 1))
        self.curvature_matrix = np.zeros((1, 1))
        self.curvature_reg_matrix = np.zeros((1, 1))
        self.solution_vector = np.zeros((1))

    @property
    def reconstructed_image(self):
        return np.zeros((1, 1))
