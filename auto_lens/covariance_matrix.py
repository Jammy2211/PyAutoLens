class CovarianceMatrixGenerator(object):
    def __init__(self, pixel_maps, noise_vector):
        self.pixel_maps = pixel_maps
        self.noise_vector = noise_vector

    def calculate_covariance(self, source_index_a, source_index_b):
        mapping_dict_1 = self.pixel_maps[source_index_a]
        mapping_dict_2 = self.pixel_maps[source_index_b]
        return sum([mapping_dict_1[i] * mapping_dict_2[i] / self.noise_vector[i] for i in mapping_dict_1.keys() if
                    i in mapping_dict_2])


class TestCase(object):
    def test_calculate_covariance(self):
        generator = CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1}], [1, 1])

        assert generator.calculate_covariance(0, 1) == 2

    def test_no_covariance(self):
        generator = CovarianceMatrixGenerator([{1: 3}, {0: 1}], [1, 1])

        assert generator.calculate_covariance(0, 1) == 0

    def test_variable_noise(self):
        generator = CovarianceMatrixGenerator([{0: 2, 1: 3}, {0: 1, 1: 1}], [2, 3])

        assert generator.calculate_covariance(0, 1) == 2
