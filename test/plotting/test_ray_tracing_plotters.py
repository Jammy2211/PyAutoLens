import itertools
import os
import shutil
from functools import wraps

import pytest
import numpy as np

from autolens import conf
from autolens.imaging import mask as msk
from autolens.plotting import ray_tracing_plotters
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp
from autolens.lensing import galaxy as g
from autolens.lensing import ray_tracing

@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../config/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")


@pytest.fixture(name='galaxy_plotter_path')
def test_galaxy_plotter_setup():
    galaxy_plotter_path = "{}/../test_files/plotting/galaxy/".format(os.path.dirname(os.path.realpath(__file__)))

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
    return msk.ImagingGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)

@pytest.fixture(name='tracer')
def test_tracer(galaxy_light, galaxy_mass, grids):
    return ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass, galaxy_light], source_galaxies=[galaxy_light],
                                               image_plane_grids=grids)

def

def test__intensities_is_output(galaxy_light, grids, galaxy_plotter_path):
    galaxy_plotters.plot_intensities(galaxy=galaxy_light, grid=grids.image,
                                      output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_intensities.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_intensities.png')

def test__individual_intensities_is_output(galaxy_light, grids, galaxy_plotter_path):
    galaxy_plotters.plot_intensities_individual(galaxy=galaxy_light, grid=grids.image,
                                      output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_individual_intensities.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_individual_intensities.png')

def test__surface_density_is_output(galaxy_mass, grids, galaxy_plotter_path):
    galaxy_plotters.plot_surface_density(galaxy=galaxy_mass, grid=grids.image,
                                          output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_surface_density.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_surface_density.png')

def test__potential_is_output(galaxy_mass, grids, galaxy_plotter_path):
    galaxy_plotters.plot_potential(galaxy=galaxy_mass, grid=grids.image,
                                    output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_potential.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_potential.png')

def test__deflections_is_output(galaxy_mass, grids, galaxy_plotter_path):
    galaxy_plotters.plot_deflections(galaxy=galaxy_mass, grid=grids.image,
                                      output_path=galaxy_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_plotter_path + 'galaxy_deflections.png')
    os.remove(path=galaxy_plotter_path + 'galaxy_deflections.png')