import itertools
import os
import shutil
from functools import wraps

import pytest
import numpy as np

from autolens import conf
from autolens.imaging import mask as msk
from autolens.plotting import plane_plotters
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.lensing import galaxy as g
from autolens.lensing import plane as pl


@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../config/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")


@pytest.fixture(name='plane_plotter_path')
def test_plane_plotter_setup():
    plane_plotter_path = "{}/../test_files/plotting/plane/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(plane_plotter_path):
        shutil.rmtree(plane_plotter_path)

    os.mkdir(plane_plotter_path)

    return plane_plotter_path


@pytest.fixture(name='galaxy_light')
def test_galaxy_light():
    return g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))


@pytest.fixture(name='galaxy_mass')
def test_galaxy_mass():
    return g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0))


@pytest.fixture(name='grids')
def test_grids():
    return msk.ImagingGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)

@pytest.fixture(name='plane')
def test_plane(galaxy_light, grids):
    return pl.Plane(galaxies=[galaxy_light], grids=grids)


def test__plane_grid_is_output(plane, plane_plotter_path):
    plane_plotters.plot_plane_grid(plane=plane, output_path=plane_plotter_path, output_format='png')
    assert os.path.isfile(path=plane_plotter_path + 'plane_grid.png')
    os.remove(path=plane_plotter_path + 'plane_grid.png')