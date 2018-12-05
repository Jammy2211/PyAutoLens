import os
import shutil

import pytest

from autofit import conf
from autolens.data.array import grids
from autolens.model.galaxy.plotters import galaxy_plotters
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.model.galaxy import galaxy as g


@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")


@pytest.fixture(name='galaxy_plotter_path')
def test_galaxy_plotter_setup():
    galaxy_plotter_path = "{}/../../../test_files/plotting/model_galaxy/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(galaxy_plotter_path):
        shutil.rmtree(galaxy_plotter_path)

    os.mkdir(galaxy_plotter_path)

    return galaxy_plotter_path


@pytest.fixture(name='galaxy_light')
def test_galaxy_light():
    return g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))


@pytest.fixture(name='galaxy_mass')
def test_galaxy_mass():
    return g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0))


@pytest.fixture(name='grids')
def test_grids():
    return grids.DataGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)


def test__intensities_is_output(galaxy_light, grids, galaxy_plotter_path):
    galaxy_plotters.plot_intensities(galaxy=galaxy_light, grid=grids.regular,
                                     output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_intensities.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_intensities.png')

def test__surface_density_is_output(galaxy_mass, grids, galaxy_plotter_path):
    galaxy_plotters.plot_surface_density(galaxy=galaxy_mass, grid=grids.regular,
                                         output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_surface_density.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_surface_density.png')

def test__potential_is_output(galaxy_mass, grids, galaxy_plotter_path):
    galaxy_plotters.plot_potential(galaxy=galaxy_mass, grid=grids.regular,
                                   output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_potential.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_potential.png')

def test__deflections_y_is_output(galaxy_mass, grids, galaxy_plotter_path):
    galaxy_plotters.plot_deflections_y(galaxy=galaxy_mass, grid=grids.regular,
                                       output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_deflections_y.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_deflections_y.png')

def test__deflections_x_is_output(galaxy_mass, grids, galaxy_plotter_path):
    galaxy_plotters.plot_deflections_x(galaxy=galaxy_mass, grid=grids.regular,
                                       output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_deflections_x.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_deflections_x.png')

def test__individual_intensities_is_output(galaxy_light, grids, galaxy_plotter_path):
    galaxy_plotters.plot_intensities_subplot(galaxy=galaxy_light, grid=grids.regular,
                                             output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_individual_intensities.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_individual_intensities.png')

def test__individual_surface_density_is_output(galaxy_light, grids, galaxy_plotter_path):
    galaxy_plotters.plot_surface_density_subplot(galaxy=galaxy_light, grid=grids.regular,
                                                 output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_individual_surface_density.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_individual_surface_density.png')
    
def test__individual_potential_is_output(galaxy_light, grids, galaxy_plotter_path):
    galaxy_plotters.plot_potential_subplot(galaxy=galaxy_light, grid=grids.regular,
                                           output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_individual_potential.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_individual_potential.png')
    
def test__individual_deflections_y_is_output(galaxy_light, grids, galaxy_plotter_path):
    galaxy_plotters.plot_deflections_y_subplot(galaxy=galaxy_light, grid=grids.regular,
                                               output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_individual_deflections_y.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_individual_deflections_y.png')
    
def test__individual_deflections_x_is_output(galaxy_light, grids, galaxy_plotter_path):
    galaxy_plotters.plot_intensities_subplot(galaxy=galaxy_light, grid=grids.regular,
                                             output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_individual_intensities.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_individual_intensities.png')