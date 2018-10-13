from os import path

import pytest

from autolens import conf
from autolens import exc
from autolens.profiles import light_profiles

directory = path.dirname(path.realpath(__file__))


class TestCase(object):
    def test_is_config(self):
        assert conf.is_config()


class MockClass(object):
    pass


@pytest.fixture(name="label_config")
def make_label_config():
    return conf.LabelConfig("{}/test_files/configs/config/label.ini".format(directory))


class TestLabel(object):
    def test_basic(self, label_config):
        assert label_config.label("centre_0") == "x"
        assert label_config.label("redshift") == "z"

    def test_escaped(self, label_config):
        assert label_config.label("gamma") == r"\gamma"
        assert label_config.label("contribution_factor") == r"\omega0"

    def test_subscript(self, label_config):
        assert label_config.subscript(light_profiles.EllipticalLP) == "l"

    def test_inheritance(self, label_config):
        assert label_config.subscript(light_profiles.EllipticalGaussian) == "l"

    def test_exception(self, label_config):
        with pytest.raises(exc.PriorException):
            label_config.subscript(MockClass)
