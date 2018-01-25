def calculate_covariance(mapping_dict_1, mapping_dict_2):
    return sum([mapping_dict_1[i] * mapping_dict_2[i] for i in mapping_dict_1.keys() if i in mapping_dict_2])


class TestCase(object):
    def test_calculate_covariance(self):
        d1 = {1: 2, 2: 3}
        d2 = {1: 1}

        assert calculate_covariance(d1, d2) == 2

    def test_no_covariance(self):
        d1 = {2: 3}
        d2 = {1: 1}

        assert calculate_covariance(d1, d2) == 0
