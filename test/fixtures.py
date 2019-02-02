import os

import pytest
from matplotlib import pyplot

from autofit import conf


@pytest.fixture(name='general_config', autouse=True)
def make_general_config():
    general_config_path = "{}/test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")


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
