from os import path

import pytest

from autoconf import conf

directory = path.dirname(path.realpath(__file__))


class MockClass:
    pass


@pytest.fixture(name="label_config")
def make_label_config():
    print(directory, "config")

    config = conf.Config(
        path.join(directory, "config"), output_path=path.join(directory, "output")
    )

    return config["notation"]["label"]


class TestLabel:
    def test_basic(self, label_config):
        assert label_config["label"]["centre_0"] == "y"
        assert label_config["label"]["redshift"] == "z"

    def test_escaped(self, label_config):
        assert label_config["label"]["gamma"] == r"\gamma"
        print(label_config["label"]["contribution_factor"])
        assert label_config["label"]["contribution_factor"] == r"\omega_{\rm 0}"

    def test_exception(self, label_config):
        with pytest.raises(KeyError):
            label_config["subscript"].family(MockClass)
