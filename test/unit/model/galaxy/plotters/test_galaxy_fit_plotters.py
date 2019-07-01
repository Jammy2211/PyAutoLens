from autolens.model.galaxy.plotters import galaxy_fit_plotters
from test.fixtures import *

import numpy as np

@pytest.fixture(name='galaxy_fitting_plotter_path')
def make_galaxy_fitting_plotter_setup():
    return "{}/../../../test_files/plotting/galaxy_fitting/".format(os.path.dirname(os.path.realpath(__file__)))


def test__fit_sub_plot__all_types_of_galaxy_fit(
        gal_fit_5x5_intensities, gal_fit_5x5_convergence, gal_fit_5x5_potential, gal_fit_5x5_deflections_y,
        gal_fit_5x5_deflections_x,  positions_5x5, plot_patch, galaxy_fitting_plotter_path):

    galaxy_fit_plotters.plot_fit_subplot(fit=gal_fit_5x5_intensities,
                                         should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                         positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                         output_path=galaxy_fitting_plotter_path, output_format='png')

    assert galaxy_fitting_plotter_path + 'galaxy_fit.png' in plot_patch.paths

    galaxy_fit_plotters.plot_fit_subplot(fit=gal_fit_5x5_convergence,
                                         should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                         positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                         output_path=galaxy_fitting_plotter_path, output_format='png')

    assert galaxy_fitting_plotter_path + 'galaxy_fit.png' in plot_patch.paths

    galaxy_fit_plotters.plot_fit_subplot(fit=gal_fit_5x5_potential,
                                         should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                         positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                         output_path=galaxy_fitting_plotter_path, output_format='png')

    assert galaxy_fitting_plotter_path + 'galaxy_fit.png' in plot_patch.paths


    galaxy_fit_plotters.plot_fit_subplot(fit=gal_fit_5x5_deflections_y,
                                         should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                         positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                         output_path=galaxy_fitting_plotter_path, output_format='png')

    assert galaxy_fitting_plotter_path + 'galaxy_fit.png' in plot_patch.paths

    galaxy_fit_plotters.plot_fit_subplot(fit=gal_fit_5x5_deflections_x,
                                         should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                         positions=positions_5x5, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                         output_path=galaxy_fitting_plotter_path, output_format='png')

    assert galaxy_fitting_plotter_path + 'galaxy_fit.png' in plot_patch.paths
