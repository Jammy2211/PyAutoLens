import numpy as np

from autolens.data import ccd
from autolens.data.array import mask as msk, scaled_array
from autolens.data.plotters import ccd_plotters
from test.fixtures import *


@pytest.fixture(name='ccd_plotter_path')
def make_ccd_plotter_setup():
    ccd_plotter_path = "{}/../../test_files/plotting/ccd/".format(os.path.dirname(os.path.realpath(__file__)))

    return ccd_plotter_path


@pytest.fixture(name='ccd_data')
def make_ccd():
    image = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
    noise_map = ccd.NoiseMap(array=2.0 * np.ones((3, 3)), pixel_scale=1.0)
    psf = ccd.PSF(array=3.0 * np.ones((3, 3)), pixel_scale=1.0)

    return ccd.CCDData(image=image, pixel_scale=1.0, noise_map=noise_map, psf=psf)


@pytest.fixture(name='positions')
def make_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=((3, 3)), pixel_scale=0.1, radius_arcsec=0.1)


def test__image_is_output(ccd_data, positions, mask, ccd_plotter_path, plot_patch):
    ccd_plotters.plot_image(ccd_data=ccd_data, positions=positions, mask=mask, extract_array_from_mask=True,
                            zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                            output_path=ccd_plotter_path, output_format='png')
    assert ccd_plotter_path + 'ccd_image.png' in plot_patch.paths


def test__noise_map_is_output(ccd_data, mask, ccd_plotter_path, plot_patch):
    ccd_plotters.plot_noise_map(ccd_data=ccd_data, mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                output_path=ccd_plotter_path, output_format='png')
    assert ccd_plotter_path + 'ccd_noise_map.png' in plot_patch.paths


def test__psf_is_output(ccd_data, ccd_plotter_path, plot_patch):
    ccd_plotters.plot_psf(ccd_data=ccd_data, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                          output_path=ccd_plotter_path, output_format='png')
    assert ccd_plotter_path + 'ccd_psf.png' in plot_patch.paths


def test__signal_to_noise_map_is_output(ccd_data, mask, ccd_plotter_path, plot_patch):
    ccd_plotters.plot_signal_to_noise_map(ccd_data=ccd_data, extract_array_from_mask=True, zoom_around_mask=True,
                                          mask=mask, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                          output_path=ccd_plotter_path, output_format='png')
    assert ccd_plotter_path + 'ccd_signal_to_noise_map.png' in plot_patch.paths


def test__ccd_sub_plot_output(ccd_data, general_config, ccd_plotter_path, plot_patch):
    ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                  output_path=ccd_plotter_path, output_format='png')

    assert ccd_plotter_path + 'ccd_data.png' in plot_patch.paths


def test__ccd_individuals__output_dependent_on_input(ccd_data, general_config, ccd_plotter_path, plot_patch):

    ccd_plotters.plot_ccd_individual(ccd_data=ccd_data,
                                     should_plot_image=True,
                                     should_plot_psf=True,
                                     should_plot_absolute_signal_to_noise_map=True,
                                     output_path=ccd_plotter_path, output_format='png')

    assert ccd_plotter_path + 'ccd_image.png' in plot_patch.paths

    assert not ccd_plotter_path + 'ccd_noise_map.png' in plot_patch.paths

    assert ccd_plotter_path + 'ccd_psf.png' in plot_patch.paths

    assert not ccd_plotter_path + 'ccd_signal_to_noise_map.png' in plot_patch.paths

    assert ccd_plotter_path + 'ccd_absolute_signal_to_noise_map.png' in plot_patch.paths

    assert not ccd_plotter_path + 'ccd_potential_chi_squared_map.png' in plot_patch.paths
    
    
def test__plot_ccd_for_phase(ccd_data, mask, general_config, ccd_plotter_path, plot_patch):

    ccd_plotters.plot_ccd_for_phase(ccd_data=ccd_data, mask=mask, positions=None, units='arcsec',
                                    zoom_around_mask=True, extract_array_from_mask=True,
                                    should_plot_as_subplot=True,
                                    should_plot_image=True,
                                    should_plot_noise_map=False,
                                    should_plot_psf=True,
                                    should_plot_signal_to_noise_map=False,
                                    should_plot_absolute_signal_to_noise_map=False,
                                    should_plot_potential_chi_squared_map=True,
                                    visualize_path=ccd_plotter_path)

    assert ccd_plotter_path + 'ccd_data.png' in plot_patch.paths
    assert ccd_plotter_path + 'ccd_image.png' in plot_patch.paths
    assert ccd_plotter_path + 'ccd_noise_map.png' not in plot_patch.paths
    assert ccd_plotter_path + 'ccd_psf.png' in plot_patch.paths
    assert ccd_plotter_path + 'ccd_signal_to_noise_map.png' not in plot_patch.paths
    assert ccd_plotter_path + 'ccd_absolute_signal_to_noise_map.png' not in plot_patch.paths
    assert ccd_plotter_path + 'ccd_potential_chi_squared_map.png' in plot_patch.paths