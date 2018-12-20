import os
import shutil
from astropy import cosmology as cosmo
import pytest
import numpy as np

from autofit import conf
from autolens.data.imaging import image as im
from autolens.data.array import grids, mask as msk, scaled_array
from autolens.lensing.plotters import lensing_plotter_util
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.lensing import lensing_image as li, lensing_fitters
from autolens.model.galaxy import galaxy as g
from autolens.lensing import ray_tracing


@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")

@pytest.fixture(name='lensing_plotter_util_path')
def test_lensing_plotter_util_path_setup():
    lensing_plotter_util_path = "{}/../../test_files/plotting/lensing_plotter_util/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(lensing_plotter_util_path):
        shutil.rmtree(lensing_plotter_util_path)

    os.mkdir(lensing_plotter_util_path)

    return lensing_plotter_util_path

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

    return im.Image(array=image, pixel_scale=1.0, noise_map=noise_map, psf=psf)

@pytest.fixture(name='positions')
def test_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))

@pytest.fixture(name='mask')
def test_mask():
    return msk.Mask.circular(shape=((3,3)), pixel_scale=0.1, radius_arcsec=0.1)

@pytest.fixture(name='lensing_image')
def test_lensing_image(image, mask):
    return li.LensingImage(image=image, mask=mask)

@pytest.fixture(name='fit')
def test_fit(lensing_image, galaxy_light, galaxy_mass):
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[galaxy_light],
                                                 image_plane_grid_stack=lensing_image.grid_stack, cosmology=cosmo.Planck15)
    return lensing_fitters.fit_lensing_image_with_tracer(lensing_image=lensing_image, tracer=tracer)

@pytest.fixture(name='hyper')
def make_hyper():
    class Hyper(object):

        def __init__(self):
            pass

    hyper = Hyper()

    hyper.hyper_model_image = np.ones((5))
    hyper.hyper_galaxy_images = [np.ones((5))]
    hyper.hyper_minimum_values = [0.2]

    hyper_galaxy = g.HyperGalaxy(contribution_factor=4.0, noise_factor=2.0, noise_power=3.0)
    hyper.hyper_galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=1.0), hyper_galaxy=hyper_galaxy)
    return hyper

@pytest.fixture(name='lensing_hyper_image')
def test_lensing_hyper_image(image, mask, hyper):

    return li.LensingHyperImage(image=image, mask=mask, hyper_model_image=hyper.hyper_model_image,
                                hyper_galaxy_images=hyper.hyper_galaxy_images,
                                hyper_minimum_values=hyper.hyper_minimum_values)

@pytest.fixture(name='fit_hyper')
def test_fit_hyper(lensing_hyper_image, hyper):
    tracer = ray_tracing.TracerImagePlane(lens_galaxies=[hyper.hyper_galaxy],
                                          image_plane_grid_stack=[lensing_hyper_image.grid_stack])
    return lensing_fitters.fit_lensing_image_with_tracer(lensing_image=lensing_hyper_image, tracer=tracer)

def test__image_is_output(fit, lensing_plotter_util_path):

    lensing_plotter_util.plot_image(fit=fit, output_path=lensing_plotter_util_path, output_format='png')
    assert os.path.isfile(path=lensing_plotter_util_path + 'fit_image.png')
    os.remove(path=lensing_plotter_util_path + 'fit_image.png')

def test__noise_map_is_output(fit, lensing_plotter_util_path):

    lensing_plotter_util.plot_noise_map(fit=fit, output_path=lensing_plotter_util_path,
                                                  output_format='png')
    assert os.path.isfile(path=lensing_plotter_util_path + 'fit_noise_map.png')
    os.remove(path=lensing_plotter_util_path + 'fit_noise_map.png')

def test__model_image_is_output(fit, lensing_plotter_util_path):
    lensing_plotter_util.plot_model_image(fit=fit, output_path=lensing_plotter_util_path,
                                              output_format='png')
    assert os.path.isfile(path=lensing_plotter_util_path + 'fit_model_image.png')
    os.remove(path=lensing_plotter_util_path + 'fit_model_image.png')

def test__residual_map_is_output(fit, lensing_plotter_util_path):

    lensing_plotter_util.plot_residual_map(fit=fit, output_path=lensing_plotter_util_path,
                                               output_format='png')
    assert os.path.isfile(path=lensing_plotter_util_path + 'fit_residual_map.png')
    os.remove(path=lensing_plotter_util_path + 'fit_residual_map.png')

def test__chi_squared_map_is_output(fit, lensing_plotter_util_path):
    lensing_plotter_util.plot_chi_squared_map(fit=fit, output_path=lensing_plotter_util_path, output_format='png')
    assert os.path.isfile(path=lensing_plotter_util_path + 'fit_chi_squared_map.png')
    os.remove(path=lensing_plotter_util_path + 'fit_chi_squared_map.png')
    
def test__contributions_is_output(fit_hyper, lensing_plotter_util_path):

    lensing_plotter_util.plot_contribution_maps(fit=fit_hyper, output_path=lensing_plotter_util_path, output_format='png')
    assert os.path.isfile(path=lensing_plotter_util_path + 'fit_contribution_maps.png')
    os.remove(path=lensing_plotter_util_path + 'fit_contribution_maps.png')