import os
import shutil

import pytest
import numpy as np

from autofit import conf
from autolens.data import ccd
from autolens.data.array import mask as msk, scaled_array
from autolens.data.plotters import data_plotters


@pytest.fixture(name='general_config')
def make_general_config():
    general_config_path = "{}/../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path+"general.ini")

@pytest.fixture(name='data_plotter_path')
def make_data_plotter_setup():
    data_plotter_path = "{}/../../test_files/plotting/data/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(data_plotter_path):
        shutil.rmtree(data_plotter_path)

    os.mkdir(data_plotter_path)

    return data_plotter_path

@pytest.fixture(name='image')
def make_image():
    return scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)

@pytest.fixture(name='noise_map')
def make_noise_map():
    return ccd.NoiseMap(array=2.0*np.ones((3,3)), pixel_scale=1.0)

@pytest.fixture(name='psf')
def make_psf():
    return ccd.PSF(array=3.0*np.ones((3,3)), pixel_scale=1.0)

@pytest.fixture(name='positions')
def make_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))

@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=((3,3)), pixel_scale=0.1, radius_arcsec=0.1)


def test__image_is_output(image, positions, mask, data_plotter_path):

    data_plotters.plot_image(image=image, positions=positions, mask=mask, extract_array_from_mask=True,
                             zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                             output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path+'image.png')
    os.remove(path=data_plotter_path+'image.png')

def test__noise_map_is_output(noise_map, mask, data_plotter_path):

    data_plotters.plot_noise_map(noise_map=noise_map, mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                 cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                 output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path+'noise_map.png')
    os.remove(path=data_plotter_path+'noise_map.png')

def test__psf_is_output(psf, data_plotter_path):

    data_plotters.plot_psf(psf=psf, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                           output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path+'psf.png')
    os.remove(path=data_plotter_path+'psf.png')

def test__signal_to_noise_map_is_output(image, noise_map, mask, data_plotter_path):

    data_plotters.plot_signal_to_noise_map(signal_to_noise_map=image/noise_map, mask=mask, extract_array_from_mask=True,
                                           zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                           output_path=data_plotter_path, output_format='png')
    assert os.path.isfile(path=data_plotter_path+'signal_to_noise_map.png')
    os.remove(path=data_plotter_path+'signal_to_noise_map.png')