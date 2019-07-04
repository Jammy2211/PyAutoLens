import numpy as np

from autolens.data import ccd
from autolens.data.array import mask as msk, scaled_array
from autolens.data.plotters import data_plotters
from test.fixtures import *


@pytest.fixture(name='data_plotter_path')
def make_data_plotter_setup():
    data_plotter_path = "{}/../../test_files/plotting/data/".format(os.path.dirname(os.path.realpath(__file__)))
    return data_plotter_path

def test__all_data_types_are_output(
        image_5x5, noise_map_5x5, psf_3x3, positions_5x5, mask_5x5, data_plotter_path, plot_patch):
    
    data_plotters.plot_image(
        image=image_5x5, positions=positions_5x5, mask=mask_5x5, extract_array_from_mask=True,
        zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=data_plotter_path, output_format='png')
    
    assert data_plotter_path + 'image.png' in plot_patch.paths

    data_plotters.plot_noise_map(
        noise_map=noise_map_5x5, mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=data_plotter_path, output_format='png')

    assert data_plotter_path + 'noise_map.png' in plot_patch.paths

    data_plotters.plot_psf(
        psf=psf_3x3, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=data_plotter_path, output_format='png')

    assert data_plotter_path + 'psf.png' in plot_patch.paths

    data_plotters.plot_signal_to_noise_map(
        signal_to_noise_map=image_5x5 / noise_map_5x5, mask=mask_5x5, extract_array_from_mask=True,
        zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=data_plotter_path, output_format='png')

    assert data_plotter_path + 'signal_to_noise_map.png' in plot_patch.paths

    data_plotters.plot_absolute_signal_to_noise_map(
        absolute_signal_to_noise_map=image_5x5 / noise_map_5x5, mask=mask_5x5,
        extract_array_from_mask=True,
        zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=data_plotter_path, output_format='png')

    assert data_plotter_path + 'absolute_signal_to_noise_map.png' in plot_patch.paths
    
    data_plotters.plot_potential_chi_squared_map(
        potential_chi_squared_map=image_5x5 / noise_map_5x5, mask=mask_5x5,
        extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=data_plotter_path, output_format='png')

    assert data_plotter_path + 'potential_chi_squared_map.png' in plot_patch.paths
