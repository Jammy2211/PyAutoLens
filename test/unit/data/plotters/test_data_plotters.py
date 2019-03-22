import numpy as np

from autolens.data import ccd
from autolens.data.array import mask as msk, scaled_array
from autolens.data.plotters import data_plotters
from test.fixtures import *


@pytest.fixture(name='data_plotter_path')
def make_data_plotter_setup():
    data_plotter_path = "{}/../../test_files/plotting/data/".format(os.path.dirname(os.path.realpath(__file__)))
    return data_plotter_path


@pytest.fixture(name='image')
def make_image():
    return scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)


@pytest.fixture(name='noise_map')
def make_noise_map():
    return ccd.NoiseMap(array=2.0 * np.ones((3, 3)), pixel_scale=1.0)


@pytest.fixture(name='psf')
def make_psf():
    return ccd.PSF(array=3.0 * np.ones((3, 3)), pixel_scale=1.0)


@pytest.fixture(name='positions')
def make_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=((3, 3)), pixel_scale=0.1, radius_arcsec=0.1)


def test__image_is_output(image, positions, mask, data_plotter_path, plot_patch):
    data_plotters.plot_image(image=image, positions=positions, mask=mask, extract_array_from_mask=True,
                             zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                             output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'image.png' in plot_patch.paths


def test__noise_map_is_output(noise_map, mask, data_plotter_path, plot_patch):
    data_plotters.plot_noise_map(noise_map=noise_map, mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                 cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                 output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'noise_map.png' in plot_patch.paths


def test__psf_is_output(psf, data_plotter_path, plot_patch):
    data_plotters.plot_psf(psf=psf, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                           output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'psf.png' in plot_patch.paths


def test__signal_to_noise_map_is_output(image, noise_map, mask, data_plotter_path, plot_patch):
    data_plotters.plot_signal_to_noise_map(signal_to_noise_map=image / noise_map, mask=mask,
                                           extract_array_from_mask=True,
                                           zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                           output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'signal_to_noise_map.png' in plot_patch.paths


def test__absolute_signal_to_noise_map_is_output(image, noise_map, mask, data_plotter_path, plot_patch):
    data_plotters.plot_absolute_signal_to_noise_map(absolute_signal_to_noise_map=image / noise_map, mask=mask,
                                           extract_array_from_mask=True,
                                           zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                           output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'absolute_signal_to_noise_map.png' in plot_patch.paths
    
    
def test__potential_chi_squared_map_is_output(image, noise_map, mask, data_plotter_path, plot_patch):
    
    data_plotters.plot_potential_chi_squared_map(potential_chi_squared_map=image / noise_map, mask=mask,
                                           extract_array_from_mask=True,
                                           zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                           output_path=data_plotter_path, output_format='png')
    assert data_plotter_path + 'potential_chi_squared_map.png' in plot_patch.paths
