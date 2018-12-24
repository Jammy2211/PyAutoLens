import os
import shutil
from astropy import cosmology as cosmo
import pytest
import numpy as np

from autofit import conf
from autolens.data import ccd as im
from autolens.data.array import grids, mask as msk, scaled_array
from autolens.lens.plotters import lens_plotter_util
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.lens import lens_data as li, lens_fit
from autolens.model.galaxy import galaxy as g
from autolens.lens import ray_tracing


@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")

@pytest.fixture(name='lens_plotter_util_path')
def test_lens_plotter_util_path_setup():
    lens_plotter_util_path = "{}/../../test_files/plotting/lens_plotter_util/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(lens_plotter_util_path):
        shutil.rmtree(lens_plotter_util_path)

    os.mkdir(lens_plotter_util_path)

    return lens_plotter_util_path

@pytest.fixture(name='galaxy_light')
def test_galaxy_light():
    return g.Galaxy(light=lp.EllipticalSersic(intensity=1.0), redshift=2.0)

@pytest.fixture(name='galaxy_mass')
def test_galaxy_mass():
    return g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)

@pytest.fixture(name='grid_stack')
def test_grid_stack():
    return grids.GridStack.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)

@pytest.fixture(name='image')
def test_image():

    image = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
    noise_map = im.NoiseMap(array=2.0*np.ones((3,3)), pixel_scale=1.0)
    psf = im.PSF(array=3.0*np.ones((1,1)), pixel_scale=1.0)

    return im.CCDData(image=image, pixel_scale=1.0, noise_map=noise_map, psf=psf)

@pytest.fixture(name='positions')
def test_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))

@pytest.fixture(name='mask')
def test_mask():
    return msk.Mask.circular(shape=((3,3)), pixel_scale=0.1, radius_arcsec=0.1)

@pytest.fixture(name='lens_data')
def test_lens_image(image, mask):
    return li.LensData(ccd_data=image, mask=mask)

@pytest.fixture(name='fit')
def test_fit(lens_data, galaxy_light, galaxy_mass):
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[galaxy_light],
                                                 image_plane_grid_stack=lens_data.grid_stack, cosmology=cosmo.Planck15)
    return lens_fit.fit_lens_image_with_tracer(lens_data=lens_data, tracer=tracer)

@pytest.fixture(name='hyper')
def make_hyper():
    class Hyper(object):

        def __init__(self):
            pass

    hyper = Hyper()

    hyper.hyper_model_image = np.array([[3.0, 5.0, 7.0],
                                        [9.0, 8.0, 1.0],
                                        [4.0, 0.0, 9.0]])
    hyper.hyper_galaxy_images = [np.array([[1.0, 3.0, 5.0],
                                           [7.0, 9.0, 8.0],
                                           [6.0, 4.0, 0.0]])]
    hyper.hyper_minimum_values = [0.2, 0.8]

    hyper_galaxy = g.HyperGalaxy(contribution_factor=4.0, noise_factor=2.0, noise_power=3.0)
    hyper.hyper_galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=1.0), hyper_galaxy=hyper_galaxy)
    return hyper

@pytest.fixture(name='lens_hyper_image')
def test_lens_hyper_image(image, mask, hyper):

    return li.LensDataHyper(ccd_data=image, mask=mask, hyper_model_image=hyper.hyper_model_image,
                            hyper_galaxy_images=hyper.hyper_galaxy_images,
                            hyper_minimum_values=hyper.hyper_minimum_values)

@pytest.fixture(name='fit_hyper')
def test_fit_hyper(lens_hyper_image, hyper):
    tracer = ray_tracing.TracerImagePlane(lens_galaxies=[hyper.hyper_galaxy],
                                          image_plane_grid_stack=lens_hyper_image.grid_stack)
    return lens_fit.fit_lens_image_with_tracer(lens_data=lens_hyper_image, tracer=tracer)

def test__image_is_output(fit, lens_plotter_util_path):

    lens_plotter_util.plot_image(fit=fit, output_path=lens_plotter_util_path, output_format='png')
    assert os.path.isfile(path=lens_plotter_util_path + 'fit_image.png')
    os.remove(path=lens_plotter_util_path + 'fit_image.png')

def test__noise_map_is_output(fit, lens_plotter_util_path):

    lens_plotter_util.plot_noise_map(fit=fit, output_path=lens_plotter_util_path,
                                     output_format='png')
    assert os.path.isfile(path=lens_plotter_util_path + 'fit_noise_map.png')
    os.remove(path=lens_plotter_util_path + 'fit_noise_map.png')

def test__model_image_is_output(fit, lens_plotter_util_path):
    lens_plotter_util.plot_model_data(fit=fit, output_path=lens_plotter_util_path,
                                      output_format='png')
    assert os.path.isfile(path=lens_plotter_util_path + 'fit_model_image.png')
    os.remove(path=lens_plotter_util_path + 'fit_model_image.png')

def test__residual_map_is_output(fit, lens_plotter_util_path):

    lens_plotter_util.plot_residual_map(fit=fit, output_path=lens_plotter_util_path,
                                        output_format='png')
    assert os.path.isfile(path=lens_plotter_util_path + 'fit_residual_map.png')
    os.remove(path=lens_plotter_util_path + 'fit_residual_map.png')

def test__chi_squared_map_is_output(fit, lens_plotter_util_path):
    lens_plotter_util.plot_chi_squared_map(fit=fit, output_path=lens_plotter_util_path, output_format='png')
    assert os.path.isfile(path=lens_plotter_util_path + 'fit_chi_squared_map.png')
    os.remove(path=lens_plotter_util_path + 'fit_chi_squared_map.png')
    
def test__contribution_map_is_output(fit_hyper, lens_plotter_util_path):

    lens_plotter_util.plot_contribution_maps(fit=fit_hyper, output_path=lens_plotter_util_path,
                                             output_format='png')
    assert os.path.isfile(path=lens_plotter_util_path + 'fit_contribution_maps.png')
    os.remove(path=lens_plotter_util_path + 'fit_contribution_maps.png')