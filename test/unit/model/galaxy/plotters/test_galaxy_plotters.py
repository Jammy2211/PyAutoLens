
from autolens.model.galaxy.plotters import galaxy_plotters
from test.fixtures import *

import numpy as np

@pytest.fixture(name='galaxy_plotter_path')
def make_galaxy_plotter_setup():
    return "{}/../../../test_files/plotting/model_galaxy/".format(os.path.dirname(os.path.realpath(__file__)))


def test__individual_images_are_output(
        gal_x1_lp_x1_mp, grid_stack_5x5, mask_5x5, positions_5x5, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_intensities(
        galaxy=gal_x1_lp_x1_mp, grid=grid_stack_5x5.regular,
        mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_intensities.png' in plot_patch.paths
    
    galaxy_plotters.plot_convergence(
        galaxy=gal_x1_lp_x1_mp, grid=grid_stack_5x5.regular,
        mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_convergence.png' in plot_patch.paths

    galaxy_plotters.plot_potential(
        galaxy=gal_x1_lp_x1_mp, grid=grid_stack_5x5.regular,
        mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=galaxy_plotter_path, output_format='png')

    assert galaxy_plotter_path + 'galaxy_potential.png' in plot_patch.paths

    galaxy_plotters.plot_deflections_y(
        galaxy=gal_x1_lp_x1_mp, grid=grid_stack_5x5.regular,
        mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_deflections_y.png' in plot_patch.paths
    
    galaxy_plotters.plot_deflections_x(
        galaxy=gal_x1_lp_x1_mp, grid=grid_stack_5x5.regular,
        mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=galaxy_plotter_path, output_format='png')

    assert galaxy_plotter_path + 'galaxy_deflections_x.png' in plot_patch.paths


def test__individual_galaxy_quantities__all_are_output(
        gal_x1_lp_x1_mp, grid_stack_5x5, mask_5x5, positions_5x5, galaxy_plotter_path, plot_patch):
    
    galaxy_plotters.plot_intensities_subplot(
        galaxy=gal_x1_lp_x1_mp, grid=grid_stack_5x5.regular,
        mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_individual_intensities.png' in plot_patch.paths

    galaxy_plotters.plot_convergence_subplot(
        galaxy=gal_x1_lp_x1_mp, grid=grid_stack_5x5.regular,
        mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_individual_convergence.png' in plot_patch.paths
    
    galaxy_plotters.plot_potential_subplot(
        galaxy=gal_x1_lp_x1_mp, grid=grid_stack_5x5.regular,
        mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_individual_potential.png' in plot_patch.paths
    
    galaxy_plotters.plot_deflections_y_subplot(
        galaxy=gal_x1_lp_x1_mp, grid=grid_stack_5x5.regular,
        mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=galaxy_plotter_path, output_format='png')
    
    assert galaxy_plotter_path + 'galaxy_individual_deflections_y.png' in plot_patch.paths
    
    galaxy_plotters.plot_intensities_subplot(
        galaxy=gal_x1_lp_x1_mp, grid=grid_stack_5x5.regular,
        mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=galaxy_plotter_path, output_format='png')

    assert galaxy_plotter_path + 'galaxy_individual_intensities.png' in plot_patch.paths
