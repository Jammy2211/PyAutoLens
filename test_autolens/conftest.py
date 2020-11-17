from os import path
from os.path import dirname, realpath

import pytest
from matplotlib import pyplot

from autofit import conf

directory = dirname(realpath(__file__))


@pytest.fixture(name="config", autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "config"), path.join(directory, "pipeline", "output")
    )
    return conf.instance


class PlotPatch:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    return plot_patch
