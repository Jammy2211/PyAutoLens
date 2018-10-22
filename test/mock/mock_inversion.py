import numpy as np

class MockPixelization(object):

    def __init__(self, value):
        self.value = value

    # noinspection PyUnusedLocal,PyShadowingNames
    def mapper_from_grids_and_borders(self, grids, borders):
        return self.value

class MockRegularization(object):

    def __init__(self, value):
        self.value = value

class MockMapper(object):

    def __init__(self):
        self.mapping_matrix = np.ones((1, 1))
        self.regularization_matrix = np.ones((1, 1))

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