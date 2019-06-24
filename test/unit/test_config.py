from os import path

import pytest
import autofit as af

from autolens.model.profiles import light_profiles

directory = path.dirname(path.realpath(__file__))


class MockClass(object):
    pass


@pytest.fixture(name="label_config")
def make_label_config():
    return af.conf.instance.label


class TestLabel(object):
    def test_basic(self, label_config):
        assert label_config.label("centre_0") == "x"
        assert label_config.label("redshift") == "z"

    def test_escaped(self, label_config):
        assert label_config.label("gamma") == r"\gamma"
        assert label_config.label("contribution_factor") == r"\omega0"

    def test_subscript(self, label_config):
        assert label_config.subscript(light_profiles.EllipticalLightProfile) == "l"

    def test_inheritance(self, label_config):
        assert label_config.subscript(light_profiles.EllipticalGaussian) == "l"

    def test_exception(self, label_config):
        with pytest.raises(af.exc.PriorException):
            label_config.subscript(MockClass)
