import os

import pytest
from matplotlib import pyplot

import autofit as af


@pytest.fixture(name='general_config', autouse=True)
def make_general_config():
    general_config_path = "{}/test_files/config/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    af.conf.instance.general = af.conf.NamedConfig(general_config_path + "general.ini")


class PlotPatch(object):
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, 'savefig', plot_patch)
    return plot_patch
