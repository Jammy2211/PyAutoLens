import os
import shutil

import pytest

from autofit import conf
from autolens.data.imaging import image as im
from autolens.data.array import grids, mask as msk, scaled_array
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion.plotters import mapper_plotters

import numpy as np

@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")

@pytest.fixture(name='mapper_plotter_path')
def test_mapper_plotter_setup():
    galaxy_plotter_path = "{}/../../../test_files/plotting/mapper/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(galaxy_plotter_path):
        shutil.rmtree(galaxy_plotter_path)

    os.mkdir(galaxy_plotter_path)

    return galaxy_plotter_path

@pytest.fixture(name='image')
def test_image():

    image = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
    noise_map = im.NoiseMap(array=2.0*np.ones((3,3)), pixel_scale=1.0)
    psf = im.PSF(array=3.0*np.ones((3,3)), pixel_scale=1.0)

    return im.Image(array=image, pixel_scale=1.0, noise_map=noise_map, psf=psf)

@pytest.fixture(name='mask')
def test_mask():
    return msk.Mask.circular(shape=((3,3)), pixel_scale=0.1, radius_arcsec=0.1)

@pytest.fixture(name='galaxy_light')
def test_galaxy_light():
    return g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))


@pytest.fixture(name='galaxy_mass')
def test_galaxy_mass():
    return g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0))

@pytest.fixture(name='grids')
def test_grids():
    return grids.DataGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)

@pytest.fixture(name='border')
def test_border(mask):
    return grids.RegularGridBorder.from_mask(mask=mask)

@pytest.fixture(name='rectangular_pixelization')
def test_rectangular_pixelization():
    return pix.Rectangular(shape=(25, 25))

@pytest.fixture(name='rectangular_mapper')
def test_rectangular_mapper(rectangular_pixelization, grids, border):
    return rectangular_pixelization.mapper_from_grids_and_border(grids=grids, border=border)

def test__image_and_rectangular_mapper_is_output(image, rectangular_mapper, mapper_plotter_path):

    mapper_plotters.plot_image_and_mapper(image=image, mapper=rectangular_mapper, should_plot_centres=True,
                                          should_plot_grid=True,
                                          image_pixels=[[0, 1, 2], [3]], source_pixels=[[1, 2], [0]],
                                          output_path=mapper_plotter_path, output_format='png')
    assert os.path.isfile(path=mapper_plotter_path + 'image_and_mapper.png')
    os.remove(path=mapper_plotter_path + 'image_and_mapper.png')

def test__rectangular_mapper_is_output(rectangular_mapper, mapper_plotter_path):

    mapper_plotters.plot_mapper(mapper=rectangular_mapper, should_plot_centres=True, should_plot_grid=True,
                                image_pixels=[[0, 1, 2], [3]], source_pixels=[[1, 2], [0]],
                                output_path=mapper_plotter_path, output_filename='rectangular_mapper',
                                output_format='png')
    assert os.path.isfile(path=mapper_plotter_path + 'rectangular_mapper.png')
    os.remove(path=mapper_plotter_path + 'rectangular_mapper.png')