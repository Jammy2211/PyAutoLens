import os
import shutil

import pytest
import numpy as np

from autofit import conf
from autolens.data.imaging import image as im
from autolens.data.array import mask as msk, scaled_array
from autolens.data.imaging.plotters import imaging_plotters


@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path+"general.ini")

@pytest.fixture(name='imaging_plotter_path')
def test_imaging_plotter_setup():
    imaging_plotter_path = "{}/../../../test_files/plotting/imaging/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(imaging_plotter_path):
        shutil.rmtree(imaging_plotter_path)

    os.mkdir(imaging_plotter_path)

    return imaging_plotter_path

@pytest.fixture(name='image')
def test_image():

    image = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
    noise_map = im.NoiseMap(array=2.0*np.ones((3,3)), pixel_scale=1.0)
    psf = im.PSF(array=3.0*np.ones((3,3)), pixel_scale=1.0)

    return im.Image(array=image, pixel_scale=1.0, noise_map=noise_map, psf=psf)

@pytest.fixture(name='positions')
def test_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))

@pytest.fixture(name='mask')
def test_mask():
    return msk.Mask.circular(shape=((3,3)), pixel_scale=0.1, radius_arcsec=0.1)


def test__image_sub_plot_output_dependent_on_config(image, general_config, imaging_plotter_path):

    imaging_plotters.plot_image_subplot(image=image, output_path=imaging_plotter_path, output_format='png')

    assert os.path.isfile(path=imaging_plotter_path+'regular.png')
    os.remove(path=imaging_plotter_path+'regular.png')

def test__image_individuals__output_dependent_on_config(image, general_config, imaging_plotter_path):

    imaging_plotters.plot_image_individual(image=image, output_path=imaging_plotter_path, output_format='png')

    assert os.path.isfile(path=imaging_plotter_path+'observed_image.png')
    os.remove(path=imaging_plotter_path+'observed_image.png')

    assert not os.path.isfile(path=imaging_plotter_path+'noise_map_.png')

    assert os.path.isfile(path=imaging_plotter_path+'psf.png')
    os.remove(path=imaging_plotter_path+'psf.png')

    assert not os.path.isfile(path=imaging_plotter_path+'signal_to_noise_map.png')

def test__image_is_output(image, positions, mask, imaging_plotter_path):

    imaging_plotters.plot_image(image=image, positions=positions, mask=mask, output_path=imaging_plotter_path,
                                output_format='png')
    assert os.path.isfile(path=imaging_plotter_path+'observed_image.png')
    os.remove(path=imaging_plotter_path+'observed_image.png')

def test__noise_map_is_output(image, mask, imaging_plotter_path):

    imaging_plotters.plot_noise_map(image=image, mask=mask, output_path=imaging_plotter_path, output_format='png')
    assert os.path.isfile(path=imaging_plotter_path+'noise_map_.png')
    os.remove(path=imaging_plotter_path+'noise_map_.png')

def test__psf_is_output(image, imaging_plotter_path):

    imaging_plotters.plot_psf(image=image, output_path=imaging_plotter_path, output_format='png')
    assert os.path.isfile(path=imaging_plotter_path+'psf.png')
    os.remove(path=imaging_plotter_path+'psf.png')

def test__signal_to_noise_map_is_output(image, mask, imaging_plotter_path):

    imaging_plotters.plot_signal_to_noise_map(image=image, mask=mask, output_path=imaging_plotter_path,
                                              output_format='png')
    assert os.path.isfile(path=imaging_plotter_path+'signal_to_noise_map.png')
    os.remove(path=imaging_plotter_path+'signal_to_noise_map.png')