from os import path

import pytest

import autofit as af
import autolens as al

directory = path.dirname(path.realpath(__file__))


class MockClass:
    pass


@pytest.fixture(name="label_config")
def make_label_config(config):
    return config["notation"]["label"]


class TestLabel:
    def test_basic(self, label_config):
        assert label_config["label"]["centre_0"] == "y"
        assert label_config["label"]["redshift"] == "z"

    def test_escaped(self, label_config):
        assert label_config["label"]["gamma"] == r"\gamma"
        assert label_config["label"]["contribution_factor"] == r"\omega0"

    def test_subscript(self, label_config):
        assert label_config["subscript"].family(al.lp.EllipticalLightProfile) == "l"

    def test_inheritance(self, label_config):
        assert label_config["subscript"].family(al.lp.EllipticalGaussian) == "l"

    def test_exception(self, label_config):
        with pytest.raises(KeyError):
            label_config["subscript"].family(MockClass)
