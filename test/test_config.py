from autolens import conf
from os import path
import pytest

directory = path.dirname(path.realpath(__file__))


class TestCase(object):
    def test_is_config(self):
        assert conf.is_config()


@pytest.fixture(name="label_config")
def make_label_config():
    return conf.LabelConfig("{}/config/label.ini".format(directory))


class TestLabel(object):
    def test_basic(self, label_config):
        assert label_config.label("centre_0") == "x"
        assert label_config.label("redshift") == "z"

    def test_escaped(self, label_config):
        assert label_config.label("gamma") == r"\gamma"
        assert label_config.label("contribution_factor") == r"\omega0"
