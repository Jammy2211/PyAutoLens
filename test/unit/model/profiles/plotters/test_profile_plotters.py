from autolens.data.array import grids, mask as msk
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.model.profiles.plotters import profile_plotters
from test.fixtures import *

import numpy as np

@pytest.fixture(name='profile_plotter_path')
def make_profile_plotter_setup():
    return "{}/../../../test_files/plotting/profiles/".format(os.path.dirname(os.path.realpath(__file__)))

@pytest.fixture(name='light_profile')
def make_light_profile():
    return lp.EllipticalSersic(intensity=1.0)

@pytest.fixture(name='mass_profile')
def make_mass_profile():
    return mp.SphericalIsothermal(einstein_radius=1.0)

@pytest.fixture(name='positions')
def make_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))

@pytest.fixture(name='grid_stack')
def make_grid_stack():
    return grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=(3, 3), pixel_scale=0.05, sub_grid_size=2)

@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=((3, 3)), pixel_scale=0.1, radius_arcsec=0.1)

def test__intensities_is_output(light_profile, grid_stack, mask, positions, profile_plotter_path, plot_patch):
    
    profile_plotters.plot_intensities(light_profile=light_profile, grid=grid_stack.regular,
                                      mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                      positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                      output_path=profile_plotter_path, output_format='png')
    
    assert profile_plotter_path + 'intensities.png' in plot_patch.paths


def test__convergence_is_output(mass_profile, grid_stack, mask, positions, profile_plotter_path, plot_patch):
    
    profile_plotters.plot_convergence(mass_profile=mass_profile, grid=grid_stack.regular,
                                      mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                      positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                      output_path=profile_plotter_path, output_format='png')
    
    assert profile_plotter_path + 'convergence.png' in plot_patch.paths


def test__potential_is_output(mass_profile, grid_stack, mask, positions, profile_plotter_path, plot_patch):
    
    profile_plotters.plot_potential(mass_profile=mass_profile, grid=grid_stack.regular,
                                    mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                    positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                    output_path=profile_plotter_path, output_format='png')
    
    assert profile_plotter_path + 'potential.png' in plot_patch.paths


def test__deflections_y_is_output(mass_profile, grid_stack, mask, positions, profile_plotter_path, plot_patch):
    
    profile_plotters.plot_deflections_y(mass_profile=mass_profile, grid=grid_stack.regular,
                                        mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                        positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                        output_path=profile_plotter_path, output_format='png')
    
    assert profile_plotter_path + 'deflections_y.png' in plot_patch.paths


def test__deflections_x_is_output(mass_profile, grid_stack, mask, positions, profile_plotter_path, plot_patch):
    
    profile_plotters.plot_deflections_x(mass_profile=mass_profile, grid=grid_stack.regular,
                                        mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                        positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                        output_path=profile_plotter_path, output_format='png')
    
    assert profile_plotter_path + 'deflections_x.png' in plot_patch.paths
