import os
import shutil
from astropy import cosmology as cosmo
import pytest
import numpy as np

from autofit import conf
from autolens.data.imaging import image as im
from autolens.data.array import grids, mask as msk, scaled_array
from autolens.lensing.plotters import lensing_fitting_plotters
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.lensing import lensing_image as li
from autolens.model.galaxy import galaxy as g
from autolens.lensing import ray_tracing
from autolens.lensing import lensing_fitting

@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")


@pytest.fixture(name='lensing_fitting_plotter_path')
def test_lensing_fitting_plotter_setup():
    galaxy_plotter_path = "{}/../../test_files/plotting/fitting/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(galaxy_plotter_path):
        shutil.rmtree(galaxy_plotter_path)

    os.mkdir(galaxy_plotter_path)

    return galaxy_plotter_path


@pytest.fixture(name='galaxy_light')
def test_galaxy_light():
    return g.Galaxy(light=lp.EllipticalSersic(intensity=1.0), redshift=2.0)

@pytest.fixture(name='galaxy_mass')
def test_galaxy_mass():
    return g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)

@pytest.fixture(name='grids')
def test_grids():
    return grids.DataGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)

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

@pytest.fixture(name='fit_lens_only')
def test_fit_lens_only(lensing_image, galaxy_light):
    tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=[lensing_image.grids],
                                          cosmology=cosmo.Planck15)
    return lensing_fitting.fit_lensing_image_with_tracer(lensing_image=lensing_image, tracer=tracer)

@pytest.fixture(name='fit_source_and_lens')
def test_fit_source_and_lens(lensing_image, galaxy_light, galaxy_mass):
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[galaxy_light],
                                               image_plane_grids=[lensing_image.grids], cosmology=cosmo.Planck15)
    return lensing_fitting.fit_lensing_image_with_tracer(lensing_image=lensing_image, tracer=tracer)

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

@pytest.fixture(name='fit_hyper_lens_only')
def test_fit_hyper_lens_only(lensing_hyper_image, hyper):
    tracer = ray_tracing.TracerImagePlane(lens_galaxies=[hyper.hyper_galaxy],
                                          image_plane_grids=[lensing_hyper_image.grids])
    return lensing_fitting.fit_lensing_image_with_tracer(lensing_image=lensing_hyper_image, tracer=tracer)


def test__fit_sub_plot_lens_only__output_dependent_on_config(fit_lens_only, general_config, lensing_fitting_plotter_path):

    lensing_fitting_plotters.plot_fitting_subplot(fit=fit_lens_only, should_plot_mask=True,
                                                  output_path=lensing_fitting_plotter_path, output_format='png')
    assert os.path.isfile(path=lensing_fitting_plotter_path + 'lensing_fit.png')
    os.remove(path=lensing_fitting_plotter_path + 'lensing_fit.png')

def test__fit_sub_plot_hyper_lens_only__output_dependent_on_config(fit_hyper_lens_only, general_config,
                                                                   lensing_fitting_plotter_path):

    lensing_fitting_plotters.plot_fitting_subplot(fit=fit_hyper_lens_only, should_plot_mask=True,
                                                  output_path=lensing_fitting_plotter_path,
                                                  output_filename='hyper_lensing_fit', output_format='png')
    assert os.path.isfile(path=lensing_fitting_plotter_path + 'hyper_lensing_fit.png')
    os.remove(path=lensing_fitting_plotter_path + 'hyper_lensing_fit.png')

def test__fit_sub_plot_source_and_lens__output_dependent_on_config(fit_source_and_lens, general_config,
                                                                   lensing_fitting_plotter_path):

    lensing_fitting_plotters.plot_fitting_subplot(fit=fit_source_and_lens, should_plot_mask=True,
                                                  output_path=lensing_fitting_plotter_path, output_format='png')
    assert os.path.isfile(path=lensing_fitting_plotter_path + 'lensing_fit.png')
    os.remove(path=lensing_fitting_plotter_path + 'lensing_fit.png')

def test__fit_individuals__lens_only__depedent_on_config(fit_lens_only, general_config, lensing_fitting_plotter_path):

    lensing_fitting_plotters.plot_fitting_individuals(fit=fit_lens_only, output_path=lensing_fitting_plotter_path,
                                                      output_format='png')

    assert os.path.isfile(path=lensing_fitting_plotter_path + 'fit_model_image.png')
    os.remove(path=lensing_fitting_plotter_path + 'fit_model_image.png')

    assert not os.path.isfile(path=lensing_fitting_plotter_path + 'fit_residuals.png')

    assert os.path.isfile(path=lensing_fitting_plotter_path + 'fit_chi_squareds.png')
    os.remove(path=lensing_fitting_plotter_path + 'fit_chi_squareds.png')

def test__fit_individuals__hyper_lens_only__depedent_on_config(fit_hyper_lens_only, general_config,
                                                               lensing_fitting_plotter_path):

    lensing_fitting_plotters.plot_fitting_individuals(fit=fit_hyper_lens_only, output_path=lensing_fitting_plotter_path,
                                                      output_format='png')

    assert os.path.isfile(path=lensing_fitting_plotter_path + 'fit_model_image.png')
    os.remove(path=lensing_fitting_plotter_path + 'fit_model_image.png')

    assert not os.path.isfile(path=lensing_fitting_plotter_path + 'fit_residuals.png')

    assert os.path.isfile(path=lensing_fitting_plotter_path + 'fit_chi_squareds.png')
    os.remove(path=lensing_fitting_plotter_path + 'fit_chi_squareds.png')

    assert not os.path.isfile(path=lensing_fitting_plotter_path + 'fit_scaled_chi_squareds.png')

    assert os.path.isfile(path=lensing_fitting_plotter_path + 'fit_scaled_noise_map.png')
    os.remove(path=lensing_fitting_plotter_path + 'fit_scaled_noise_map.png')

def test__fit_individuals__source_and_lens__depedent_on_config(fit_source_and_lens, general_config,
                                                               lensing_fitting_plotter_path):

    lensing_fitting_plotters.plot_fitting_individuals(fit=fit_source_and_lens, output_path=lensing_fitting_plotter_path,
                                                      output_format='png')

    assert os.path.isfile(path=lensing_fitting_plotter_path + 'fit_model_image.png')
    os.remove(path=lensing_fitting_plotter_path + 'fit_model_image.png')

    assert not os.path.isfile(path=lensing_fitting_plotter_path + 'fit_lens_plane_model_image.png')

    assert os.path.isfile(path=lensing_fitting_plotter_path + 'fit_source_plane_model_image.png')
    os.remove(path=lensing_fitting_plotter_path + 'fit_source_plane_model_image.png')

    assert not os.path.isfile(path=lensing_fitting_plotter_path + 'fit_residuals.png')

    assert os.path.isfile(path=lensing_fitting_plotter_path + 'fit_chi_squareds.png')
    os.remove(path=lensing_fitting_plotter_path + 'fit_chi_squareds.png')