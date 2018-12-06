import os
import shutil

import pytest
import numpy as np

from autofit import conf
from autolens.data.array import grids, mask as msk
from autolens.lensing.plotters import plane_plotters
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.lensing import plane as pl


@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")


@pytest.fixture(name='positions')
def test_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))

@pytest.fixture(name='mask')
def test_mask():
    return msk.Mask.circular(shape=((3,3)), pixel_scale=0.1, radius_mask_arcsec=0.1)

@pytest.fixture(name='plane_plotter_path')
def test_plane_plotter_setup():
    plane_plotter_path = "{}/../../test_files/plotting/plane/".format(os.path.dirname(os.path.realpath(__file__)))

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
    return grids.DataGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)

@pytest.fixture(name='plane')
def test_plane(galaxy_light, grids):
    return pl.Plane(galaxies=[galaxy_light], grids=[grids])



def test__image_plane_image_is_output(plane, mask, positions, plane_plotter_path):
    plane_plotters.plot_image_plane_image(plane=plane, mask=mask, positions=positions,
                                          output_path=plane_plotter_path, output_format='png')
    assert os.path.isfile(path=plane_plotter_path + 'plane_image_plane_image.png')
    os.remove(path=plane_plotter_path + 'plane_image_plane_image.png')

def test__plane_image_is_output(plane, positions, plane_plotter_path):
    plane_plotters.plot_plane_image(plane=plane, positions=positions, output_path=plane_plotter_path,
                                    output_format='png')
    assert os.path.isfile(path=plane_plotter_path + 'plane_image.png')
    os.remove(path=plane_plotter_path + 'plane_image.png')

def test__surface_density_is_output(plane, plane_plotter_path):
    
    plane_plotters.plot_surface_density(plane=plane, output_path=plane_plotter_path, output_format='png')
    assert os.path.isfile(path=plane_plotter_path + 'plane_surface_density.png')
    os.remove(path=plane_plotter_path + 'plane_surface_density.png')

def test__potential_is_output(plane, plane_plotter_path):
    
    plane_plotters.plot_potential(plane=plane, output_path=plane_plotter_path, output_format='png')
    assert os.path.isfile(path=plane_plotter_path + 'plane_potential.png')
    os.remove(path=plane_plotter_path + 'plane_potential.png')

def test__deflections_y_is_output(plane, plane_plotter_path):
    
    plane_plotters.plot_deflections_y(plane=plane, output_path=plane_plotter_path, output_format='png')
    assert os.path.isfile(path=plane_plotter_path + 'plane_deflections_y.png')
    os.remove(path=plane_plotter_path + 'plane_deflections_y.png')

def test__deflections_x_is_output(plane, plane_plotter_path):
    
    plane_plotters.plot_deflections_x(plane=plane, output_path=plane_plotter_path, output_format='png')
    assert os.path.isfile(path=plane_plotter_path + 'plane_deflections_x.png')
    os.remove(path=plane_plotter_path + 'plane_deflections_x.png')

def test__plane_grid_is_output(plane, plane_plotter_path):

    plane_plotters.plot_plane_grid(plane=plane, output_path=plane_plotter_path, output_format='png')
    assert os.path.isfile(path=plane_plotter_path + 'plane_grid.png')
    os.remove(path=plane_plotter_path + 'plane_grid.png')