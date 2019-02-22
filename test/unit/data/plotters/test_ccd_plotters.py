import os
import shutil

import pytest
import numpy as np

from autofit import conf
from autolens.data import ccd
from autolens.data.array import mask as msk, scaled_array
from autolens.data.plotters import ccd_plotters


@pytest.fixture(name='general_config')
def make_general_config():
    general_config_path = "{}/../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path+"general.ini")

@pytest.fixture(name='ccd_plotter_path')
def make_ccd_plotter_setup():
    ccd_plotter_path = "{}/../../test_files/plotting/ccd/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(ccd_plotter_path):
        shutil.rmtree(ccd_plotter_path)

    os.mkdir(ccd_plotter_path)

    return ccd_plotter_path

@pytest.fixture(name='ccd_data')
def make_ccd():

    image = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
    noise_map = ccd.NoiseMap(array=2.0*np.ones((3,3)), pixel_scale=1.0)
    psf = ccd.PSF(array=3.0*np.ones((3,3)), pixel_scale=1.0)

    return ccd.CCDData(image=image, pixel_scale=1.0, noise_map=noise_map, psf=psf)

@pytest.fixture(name='positions')
def make_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))

@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=((3,3)), pixel_scale=0.1, radius_arcsec=0.1)


def test__ccd_sub_plot_output(ccd_data, general_config, ccd_plotter_path):

    ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                  output_path=ccd_plotter_path, output_format='png')

    assert os.path.isfile(path=ccd_plotter_path+'ccd_data.png')
    os.remove(path=ccd_plotter_path+'ccd_data.png')

def test__ccd_individuals__output_dependent_on_input(ccd_data, general_config, ccd_plotter_path):

    ccd_plotters.plot_ccd_individual(ccd_data=ccd_data,
                                     should_plot_image=True, should_plot_psf=True,
                                     output_path=ccd_plotter_path, output_format='png')

    assert os.path.isfile(path=ccd_plotter_path+'ccd_image.png')
    os.remove(path=ccd_plotter_path+'ccd_image.png')

    assert not os.path.isfile(path=ccd_plotter_path+'ccd_noise_map.png')

    assert os.path.isfile(path=ccd_plotter_path+'ccd_psf.png')
    os.remove(path=ccd_plotter_path+'ccd_psf.png')

    assert not os.path.isfile(path=ccd_plotter_path+'ccd_signal_to_noise_map.png')

def test__image_is_output(ccd_data, positions, mask, ccd_plotter_path):

    ccd_plotters.plot_image(ccd_data=ccd_data, positions=positions, mask=mask, extract_array_from_mask=True,
                            zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                            output_path=ccd_plotter_path, output_format='png')
    assert os.path.isfile(path=ccd_plotter_path+'ccd_image.png')
    os.remove(path=ccd_plotter_path+'ccd_image.png')

def test__noise_map_is_output(ccd_data, mask, ccd_plotter_path):

    ccd_plotters.plot_noise_map(ccd_data=ccd_data, mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                output_path=ccd_plotter_path, output_format='png')
    assert os.path.isfile(path=ccd_plotter_path+'ccd_noise_map.png')
    os.remove(path=ccd_plotter_path+'ccd_noise_map.png')

def test__psf_is_output(ccd_data, ccd_plotter_path):

    ccd_plotters.plot_psf(ccd_data=ccd_data, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                          output_path=ccd_plotter_path, output_format='png')
    assert os.path.isfile(path=ccd_plotter_path+'ccd_psf.png')
    os.remove(path=ccd_plotter_path+'ccd_psf.png')

def test__signal_to_noise_map_is_output(ccd_data, mask, ccd_plotter_path):

    ccd_plotters.plot_signal_to_noise_map(ccd_data=ccd_data, extract_array_from_mask=True, zoom_around_mask=True,
                                          mask=mask, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                          output_path=ccd_plotter_path, output_format='png')
    assert os.path.isfile(path=ccd_plotter_path+'ccd_signal_to_noise_map.png')
    os.remove(path=ccd_plotter_path+'ccd_signal_to_noise_map.png')