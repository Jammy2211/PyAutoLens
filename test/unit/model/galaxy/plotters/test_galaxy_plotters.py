from autolens.data.array import grids, mask as msk
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy.plotters import galaxy_plotters
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.fixtures import *

import numpy as np

@pytest.fixture(name='galaxy_light')
def make_galaxy_light():
    return g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))


@pytest.fixture(name='galaxy_mass')
def make_galaxy_mass():
    return g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0))


@pytest.fixture(name='positions')
def make_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=((3, 3)), pixel_scale=0.1, radius_arcsec=0.1)

@pytest.fixture(name='grid_stack')
def make_grid_stack():
    return grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=(3, 3), pixel_scale=0.05, sub_grid_size=2)


@pytest.fixture(name='galaxy_plotter_path')
def make_galaxy_plotter_setup():
    return "{}/../../../test_files/plotting/model_galaxy/".format(os.path.dirname(os.path.realpath(__file__)))


def test__intensities_is_output(galaxy_light,grid_stack, mask, positions, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_intensities(galaxy=galaxy_light, grid=grid_stack.regular,
                                     mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                     positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                     output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_intensities.png' in plot_patch.paths


def test__convergence_is_output(galaxy_mass,grid_stack, mask, positions, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_convergence(galaxy=galaxy_mass, grid=grid_stack.regular,
                                     mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                     positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                     output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_convergence.png' in plot_patch.paths


def test__potential_is_output(galaxy_mass,grid_stack, mask, positions, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_potential(galaxy=galaxy_mass, grid=grid_stack.regular,
                                   mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                   positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                   output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_potential.png' in plot_patch.paths


def test__deflections_y_is_output(galaxy_mass,grid_stack, mask, positions, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_deflections_y(galaxy=galaxy_mass, grid=grid_stack.regular,
                                       mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                       positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                       output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_deflections_y.png' in plot_patch.paths


def test__deflections_x_is_output(galaxy_mass,grid_stack, mask, positions, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_deflections_x(galaxy=galaxy_mass, grid=grid_stack.regular,
                                       mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                       positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                       output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_deflections_x.png' in plot_patch.paths


def test__individual_intensities_is_output(galaxy_light,grid_stack, mask, positions, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_intensities_subplot(galaxy=galaxy_light, grid=grid_stack.regular,
                                             mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                             positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                             output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_individual_intensities.png' in plot_patch.paths


def test__individual_convergence_is_output(galaxy_light,grid_stack, mask, positions, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_convergence_subplot(galaxy=galaxy_light, grid=grid_stack.regular,
                                                 mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                                 positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                                 output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_individual_convergence.png' in plot_patch.paths


def test__individual_potential_is_output(galaxy_light,grid_stack, mask, positions, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_potential_subplot(galaxy=galaxy_light, grid=grid_stack.regular,
                                           mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                           positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                           output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_individual_potential.png' in plot_patch.paths


def test__individual_deflections_y_is_output(galaxy_light,grid_stack, mask, positions, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_deflections_y_subplot(galaxy=galaxy_light, grid=grid_stack.regular,
                                               mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                               positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                               output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_individual_deflections_y.png' in plot_patch.paths


def test__individual_deflections_x_is_output(galaxy_light,grid_stack, mask, positions, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_intensities_subplot(galaxy=galaxy_light, grid=grid_stack.regular,
                                             mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                             positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                             output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_individual_intensities.png' in plot_patch.paths
