from autolens import conf
from os import path


directory = path.dirname(path.realpath(__file__))


class TestCase(object):
    def test_is_config(self):
        assert conf.is_config()


class TestLabel(object):
    def test_basic(self):
        label_config = conf.NamedConfig("{}/config/label.ini".format(directory))
        assert label_config.get("label", "centre_0") == "x"
