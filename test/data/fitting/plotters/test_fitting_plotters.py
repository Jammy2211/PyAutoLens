import os
import shutil

import numpy as np
import pytest

from autofit import conf
from autolens.data.fitting import fitting_data, fitting
from autolens.model.galaxy import galaxy as g
from autolens.data.imaging import image as im
from autolens.data.array import mask as msk, scaled_array
from autolens.data.fitting.plotters import fitting_plotters
from autolens.model.profiles import light_profiles as lp


@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")


@pytest.fixture(name='fitting_plotter_path')
def test_fitting_plotter_setup():
    galaxy_plotter_path = "{}/../../../test_files/plotting/fitting/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(galaxy_plotter_path):
        shutil.rmtree(galaxy_plotter_path)

    os.mkdir(galaxy_plotter_path)

    return galaxy_plotter_path


@pytest.fixture(name='image')
def test_image():
    image = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
    noise_map = im.NoiseMap(array=2.0 * np.ones((3, 3)), pixel_scale=1.0)
    psf = im.PSF(array=3.0 * np.ones((1, 1)), pixel_scale=1.0)

    return im.Image(array=image, pixel_scale=1.0, noise_map=noise_map, psf=psf)


@pytest.fixture(name='mask')
def test_mask():
    return msk.Mask.circular(shape=((3, 3)), pixel_scale=0.1, radius_mask_arcsec=0.1)


@pytest.fixture(name='fitting_image')
def test_fitting_image(image, mask):
    return fitting_data.FittingImage(image=image, mask=mask)


@pytest.fixture(name='fit_normal')
def test_fit(fitting_image):
    return fitting.AbstractImageFit(fitting_images=[fitting_image], model_images_=[np.ones(5)])


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


@pytest.fixture(name='fitting_hyper_image')
def test_fitting_hyper_image(image, mask, hyper):
    return fitting_data.FittingHyperImage(image=image, mask=mask, hyper_model_image=hyper.hyper_model_image,
                                          hyper_galaxy_images=hyper.hyper_galaxy_images,
                                          hyper_minimum_values=hyper.hyper_minimum_values)


@pytest.fixture(name='fit_hyper')
def test_fit_hyper(fitting_hyper_image, hyper):
    return fitting.AbstractHyperImageFit(fitting_hyper_images=[fitting_hyper_image], model_images_=[np.ones(5)],
                                         hyper_galaxies=[hyper.hyper_galaxy.hyper_galaxy])


def test__model_image_is_output(fit_normal, fitting_plotter_path):
    fitting_plotters.plot_model_image(fit=fit_normal, output_path=fitting_plotter_path, output_format='png')
    assert os.path.isfile(path=fitting_plotter_path + 'fit_model_image.png')
    os.remove(path=fitting_plotter_path + 'fit_model_image.png')


def test__residuals_is_output(fit_normal, fitting_plotter_path):
    fitting_plotters.plot_residuals(fit=fit_normal, output_path=fitting_plotter_path, output_format='png')
    assert os.path.isfile(path=fitting_plotter_path + 'fit_residuals.png')
    os.remove(path=fitting_plotter_path + 'fit_residuals.png')


def test__chi_squareds_is_output(fit_normal, fitting_plotter_path):
    fitting_plotters.plot_chi_squareds(fit=fit_normal, output_path=fitting_plotter_path, output_format='png')
    assert os.path.isfile(path=fitting_plotter_path + 'fit_chi_squareds.png')
    os.remove(path=fitting_plotter_path + 'fit_chi_squareds.png')


def test__scaled_chi_squareds_is_output(fit_hyper, fitting_plotter_path):
    fitting_plotters.plot_scaled_chi_squareds(fit=fit_hyper, output_path=fitting_plotter_path,
                                              output_format='png')
    assert os.path.isfile(path=fitting_plotter_path + 'fit_scaled_chi_squareds.png')
    os.remove(path=fitting_plotter_path + 'fit_scaled_chi_squareds.png')


def test__scaled_noise_map_is_output(fit_hyper, fitting_plotter_path):
    fitting_plotters.plot_scaled_noise_map(fit=fit_hyper, output_path=fitting_plotter_path,
                                           output_format='png')
    assert os.path.isfile(path=fitting_plotter_path + 'fit_scaled_noise_map.png')
    os.remove(path=fitting_plotter_path + 'fit_scaled_noise_map.png')
