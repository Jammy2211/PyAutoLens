import os

from autoconf import conf
import pytest
from matplotlib import pyplot


@pytest.fixture(name="general_config", autouse=True)
def make_general_config():
    directory = os.path.dirname(os.path.realpath(__file__))
    config_path = f"{directory}/integration/config"
    plotting_config_path = f"{config_path}/plotting/"
    conf.instance.general = conf.NamedConfig(plotting_config_path + "general.ini")
    conf.instance.config_path = config_path


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
