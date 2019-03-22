import numpy as np

from autolens.data.array import mask as msk
from autolens.data.array import scaled_array
from autolens.model.galaxy import galaxy as g, galaxy_fit, galaxy_data as gd
from autolens.model.galaxy.plotters import galaxy_fit_plotters
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.fixtures import *

import numpy as np

@pytest.fixture(name='galaxy_fitting_plotter_path')
def make_galaxy_fitting_plotter_setup():
    return "{}/../../../test_files/plotting/galaxy_fitting/".format(os.path.dirname(os.path.realpath(__file__)))

@pytest.fixture(name='galaxy')
def make_galaxy():
    return g.Galaxy(light=lp.SphericalSersic(intensity=1.0), mass=mp.SphericalIsothermal(einstein_radius=1.0))

@pytest.fixture(name='positions')
def make_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))

@pytest.fixture(name='image')
def make_image():
    return scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=3.0)

@pytest.fixture(name='noise_map')
def make_noise_map():
    return scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=3.0)

@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=((3, 3)), pixel_scale=0.1, radius_arcsec=0.1)

@pytest.fixture(name='galaxy_data')
def make_galaxy_data(image, noise_map):
    return gd.GalaxyData(image=image, noise_map=noise_map, pixel_scale=3.0)

@pytest.fixture(name='galaxy_data_intensities')
def make_galaxy_data_intensities(galaxy_data, mask):
    return gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask, sub_grid_size=2, use_intensities=True)

@pytest.fixture(name='galaxy_data_convergence')
def make_galaxy_data_convergence(galaxy_data, mask):
    return gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask, sub_grid_size=2, use_convergence=True)

@pytest.fixture(name='galaxy_data_potential')
def make_galaxy_data_potential(galaxy_data, mask):
    return gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask, sub_grid_size=2, use_potential=True)

@pytest.fixture(name='galaxy_data_deflections_y')
def make_galaxy_data_deflections_y(galaxy_data, mask):
    return gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask, sub_grid_size=2, use_deflections_y=True)

@pytest.fixture(name='galaxy_data_deflections_x')
def make_galaxy_data_deflections_x(galaxy_data, mask):
    return gd.GalaxyFitData(galaxy_data=galaxy_data, mask=mask, sub_grid_size=2, use_deflections_x=True)

@pytest.fixture(name='fit_intensities')
def make_galaxy_fitting_intensities(galaxy_data_intensities, galaxy):
    return galaxy_fit.GalaxyFit(galaxy_data=galaxy_data_intensities, model_galaxies=[galaxy])

@pytest.fixture(name='fit_convergence')
def make_galaxy_fitting_convergence(galaxy_data_convergence, galaxy):
    return galaxy_fit.GalaxyFit(galaxy_data=galaxy_data_convergence, model_galaxies=[galaxy])

@pytest.fixture(name='fit_potential')
def make_galaxy_fitting_potential(galaxy_data_potential, galaxy):
    return galaxy_fit.GalaxyFit(galaxy_data=galaxy_data_potential, model_galaxies=[galaxy])

@pytest.fixture(name='fit_deflections_y')
def make_galaxy_fitting_deflections_y(galaxy_data_deflections_y, galaxy):
    return galaxy_fit.GalaxyFit(galaxy_data=galaxy_data_deflections_y, model_galaxies=[galaxy])

@pytest.fixture(name='fit_deflections_x')
def make_galaxy_fitting_deflections_x(galaxy_data_deflections_x, galaxy):
    return galaxy_fit.GalaxyFit(galaxy_data=galaxy_data_deflections_x, model_galaxies=[galaxy])


def test__fit_sub_plot__galaxy_intensities(fit_intensities, positions, plot_patch,
                                                                       galaxy_fitting_plotter_path):

    galaxy_fit_plotters.plot_fit_subplot(fit=fit_intensities,
                                         should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                         positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                         output_path=galaxy_fitting_plotter_path, output_format='png')

    assert galaxy_fitting_plotter_path + 'galaxy_fit.png' in plot_patch.paths


def test__fit_sub_plot__galaxy_convergence(fit_convergence, mask, positions,
                                                                           plot_patch,
                                                                           galaxy_fitting_plotter_path):

    galaxy_fit_plotters.plot_fit_subplot(fit=fit_convergence,
                                         should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                         positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                         output_path=galaxy_fitting_plotter_path, output_format='png')

    assert galaxy_fitting_plotter_path + 'galaxy_fit.png' in plot_patch.paths


def test__fit_sub_plot__galaxy_potential(fit_potential, mask, positions, plot_patch,
                                                                     galaxy_fitting_plotter_path):

    galaxy_fit_plotters.plot_fit_subplot(fit=fit_potential,
                                         should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                         positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                         output_path=galaxy_fitting_plotter_path, output_format='png')

    assert galaxy_fitting_plotter_path + 'galaxy_fit.png' in plot_patch.paths


def test__fit_sub_plot__galaxy_deflections_y_(fit_deflections_y, mask, positions, plot_patch,
                                                                         galaxy_fitting_plotter_path):

    galaxy_fit_plotters.plot_fit_subplot(fit=fit_deflections_y,
                                         should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                         positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                         output_path=galaxy_fitting_plotter_path, output_format='png')

    assert galaxy_fitting_plotter_path + 'galaxy_fit.png' in plot_patch.paths


def test__fit_sub_plot__galaxy_deflections_x(fit_deflections_x, mask, positions, plot_patch,
                                                                         galaxy_fitting_plotter_path):

    galaxy_fit_plotters.plot_fit_subplot(fit=fit_deflections_x,
                                         should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                         positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                         output_path=galaxy_fitting_plotter_path, output_format='png')

    assert galaxy_fitting_plotter_path + 'galaxy_fit.png' in plot_patch.paths
