from os import path

import pytest

from autofit import aggregator as a

directory = path.dirname(path.realpath(__file__))
aggregator_directory = "{}/test_files/aggregator".format(directory)


@pytest.fixture(name="aggregator")
def make_aggregator():
    return a.Aggregator(aggregator_directory)


def filter_phases(aggregator, folder):
    return list(filter(lambda ag: "/{}/.metadata".format(folder) in ag.file_path, aggregator.phases))[0]


@pytest.fixture(name="one")
def make_one(aggregator):
    return filter_phases(aggregator, "one")


@pytest.fixture(name="two")
def make_two(aggregator):
    return filter_phases(aggregator, "two")


@pytest.fixture(name="three")
def make_three(aggregator):
    return filter_phases(aggregator, "three")


class TestCase(object):
    def test_total_phases(self, aggregator):
        assert len(aggregator.phases) == 3

    def test_file_paths(self, one, two, three):
        assert three.file_path == "{}/three/.metadata".format(aggregator_directory)
        assert one.file_path == "{}/one/.metadata".format(aggregator_directory)
        assert two.file_path == "{}/two/.metadata".format(aggregator_directory)

    def test_attributes(self, one, two, three):
        assert one.pipeline == "pipeline_1"
        assert one.phase == "phase_1"
        assert one.data == "lens_1"

        assert two.pipeline == "pipeline_2"
        assert two.phase == "phase_1"
        assert two.data == "lens_1"

        assert three.pipeline == "pipeline_1"
        assert three.phase == "phase_2"
        assert three.data == "lens_2"

    def test_filter_phases(self, aggregator, one, two, three):
        result = aggregator.phases_with()

        assert len(result) == 3
        assert one in result
        assert two in result
        assert three in result

        result = aggregator.phases_with(pipeline="pipeline_1")
        assert len(result) == 2
        assert one in result
        assert three in result

        result = aggregator.phases_with(data="lens_2")
        assert [three] == result

        result = aggregator.phases_with(pipeline="pipeline_2", phase="phase_1")
        assert [two] == result

    def test_phase_model_results(self, one, two, three):
        assert one.model_results == "results_one"
        assert two.model_results == "results_two"
        assert three.model_results == "results_three"

    def test_header(self, one, two, three):
        assert one.header == "pipeline_1/phase_1/lens_1"
        assert two.header == "pipeline_2/phase_1/lens_1"
        assert three.header == "pipeline_1/phase_2/lens_2"

    def test_aggregator_model_results(self, aggregator):
        assert sorted(aggregator.model_results()) == sorted("pipeline_2/phase_1/lens_1\n\n"
                                                            "results_two\n\n"
                                                            "pipeline_1/phase_2/lens_2\n\n"
                                                            "results_three\n\n"
                                                            "pipeline_1/phase_1/lens_1\n\n"
                                                            "results_one")

        assert sorted(aggregator.model_results(phase="phase_1")) == sorted("pipeline_2/phase_1/lens_1\n\n"
                                                                           "results_two\n\n"
                                                                           "pipeline_1/phase_1/lens_1\n\n"
                                                                           "results_one")

    def test_nlo(self, one, two, three):
        assert one.optimizer is not None
        assert two.optimizer is not None
        assert three.optimizer is not None

        assert one.optimizer.variable.priors == two.optimizer.variable.priors

    def test_filter_optimizers(self, aggregator, one, two, three):
        result = aggregator.optimizers_with()

        assert len(result) == 3
        assert one.optimizer in result
        assert two.optimizer in result
        assert three.optimizer in result

        result = aggregator.optimizers_with(pipeline="pipeline_1")
        assert len(result) == 2
        assert one.optimizer in result
        assert three.optimizer in result
