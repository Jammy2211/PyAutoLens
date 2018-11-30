import os
import shutil

import pytest

from autofit import conf
from autolens.model.galaxy import galaxy as g
from autolens.data.array import grids
from autolens.lensing import ray_tracing
from autolens.lensing.plotters import ray_tracing_plotters
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp


@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")


@pytest.fixture(name='ray_tracing_plotter_path')
def test_ray_tracing_plotter_setup():
    galaxy_plotter_path = "{}/../../test_files/plotting/ray_tracing/".format(os.path.dirname(os.path.realpath(__file__)))

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


@pytest.fixture(name='tracer')
def test_tracer(galaxy_light, galaxy_mass, grids):
    return ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass, galaxy_light],
                                               source_galaxies=[galaxy_light],
                                               image_plane_grids=[grids])


def test__tracer_sub_plot_output_dependent_on_config(tracer, general_config, ray_tracing_plotter_path):
    ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer, output_path=ray_tracing_plotter_path,
                                                  output_format='png')

    assert os.path.isfile(path=ray_tracing_plotter_path + 'tracer_normal.png')
    os.remove(path=ray_tracing_plotter_path + 'tracer_normal.png')


def test__tracer_individuals__depedent_on_config(tracer, general_config, ray_tracing_plotter_path):
    ray_tracing_plotters.plot_ray_tracing_individual(tracer=tracer, output_path=ray_tracing_plotter_path,
                                                     output_format='png')

    assert os.path.isfile(path=ray_tracing_plotter_path + 'tracer_image_plane_image.png')
    os.remove(path=ray_tracing_plotter_path + 'tracer_image_plane_image.png')

    assert os.path.isfile(path=ray_tracing_plotter_path + 'tracer_source_plane.png')
    os.remove(path=ray_tracing_plotter_path + 'tracer_source_plane.png')

    assert not os.path.isfile(path=ray_tracing_plotter_path + 'tracer_surface_density.png')

    assert os.path.isfile(path=ray_tracing_plotter_path + 'tracer_potential.png')
    os.remove(path=ray_tracing_plotter_path + 'tracer_potential.png')

    assert not os.path.isfile(path=ray_tracing_plotter_path + 'tracer_deflections_y.png')
    assert not os.path.isfile(path=ray_tracing_plotter_path + 'tracer_deflections_x.png')


def test__image_plane_image_is_output(tracer, ray_tracing_plotter_path):
    ray_tracing_plotters.plot_image_plane_image(tracer=tracer, output_path=ray_tracing_plotter_path,
                                                output_format='png')
    assert os.path.isfile(path=ray_tracing_plotter_path + 'tracer_image_plane_image.png')
    os.remove(path=ray_tracing_plotter_path + 'tracer_image_plane_image.png')


def test__surface_density_is_output(tracer, ray_tracing_plotter_path):
    ray_tracing_plotters.plot_surface_density(tracer=tracer, output_path=ray_tracing_plotter_path, output_format='png')
    assert os.path.isfile(path=ray_tracing_plotter_path + 'tracer_surface_density.png')
    os.remove(path=ray_tracing_plotter_path + 'tracer_surface_density.png')


def test__potential_is_output(tracer, ray_tracing_plotter_path):
    ray_tracing_plotters.plot_potential(tracer=tracer, output_path=ray_tracing_plotter_path, output_format='png')
    assert os.path.isfile(path=ray_tracing_plotter_path + 'tracer_potential.png')
    os.remove(path=ray_tracing_plotter_path + 'tracer_potential.png')


def test__deflections_y_is_output(tracer, ray_tracing_plotter_path):
    ray_tracing_plotters.plot_deflections_y(tracer=tracer, output_path=ray_tracing_plotter_path, output_format='png')
    assert os.path.isfile(path=ray_tracing_plotter_path + 'tracer_deflections_y.png')
    os.remove(path=ray_tracing_plotter_path + 'tracer_deflections_y.png')


def test__deflections_x_is_output(tracer, ray_tracing_plotter_path):
    ray_tracing_plotters.plot_deflections_x(tracer=tracer, output_path=ray_tracing_plotter_path, output_format='png')
    assert os.path.isfile(path=ray_tracing_plotter_path + 'tracer_deflections_x.png')
    os.remove(path=ray_tracing_plotter_path + 'tracer_deflections_x.png')
