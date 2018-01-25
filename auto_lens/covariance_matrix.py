def calculate_covariance(mapping_dict_1, mapping_dict_2):
    return sum([mapping_dict_1[i] * mapping_dict_2[i] for i in mapping_dict_1.keys() if i in mapping_dict_2])


class CovarianceMatrixGenerator(object):
    def __init__(self, pixel_maps):
        self.pixel_maps = pixel_maps

    def calculate_covariance(self, source_index_a, source_index_b):
        mapping_dict_1 = self.pixel_maps[source_index_a]
        mapping_dict_2 = self.pixel_maps[source_index_b]
        return sum([mapping_dict_1[i] * mapping_dict_2[i] for i in mapping_dict_1.keys() if i in mapping_dict_2])


class TestCase(object):
    def test_calculate_covariance(self):
        generator = CovarianceMatrixGenerator([{1: 2, 2: 3}, {1: 1}])

        assert generator.calculate_covariance(0, 1) == 2

    def test_no_covariance(self):
        generator = CovarianceMatrixGenerator([{2: 3}, {1: 1}])

        assert generator.calculate_covariance(0, 1) == 0
    