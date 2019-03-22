from os import path

import pytest

from autolens import aggregator as a

directory = path.dirname(path.realpath(__file__))
aggregator_directory = "{}/test_files/aggregator".format(directory)


@pytest.fixture(name="aggregator")
def make_aggregator():
    return a.Aggregator(aggregator_directory)


class TestCase(object):
    def test_aggregations(self, aggregator):
        assert len(aggregator.aggregations) == 3
        assert aggregator.aggregations[0].file_path == "{}/three/.metadata".format(aggregator_directory)
        assert aggregator.aggregations[1].file_path == "{}/one/.metadata".format(aggregator_directory)
        assert aggregator.aggregations[2].file_path == "{}/two/.metadata".format(aggregator_directory)
