import os
import shutil
from astropy import cosmology as cosmo
import pytest
import numpy as np

from autofit import conf
from autolens.data.imaging import ccd as im
from autolens.data.array import grids, mask as msk, scaled_array
from autolens.lens.plotters import sensitivity_fit_plotters
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.lens import lens_image as li
from autolens.model.galaxy import galaxy as g
from autolens.lens import ray_tracing
from autolens.lens import sensitivity_fit

@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")

@pytest.fixture(name='sensitivity_fit_plotter_path')
def test_sensitivity_fit_plotter_setup():
    galaxy_plotter_path = "{}/../../test_files/plotting/fit/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(galaxy_plotter_path):
        shutil.rmtree(galaxy_plotter_path)

    os.mkdir(galaxy_plotter_path)

    return galaxy_plotter_path

@pytest.fixture(name='grid_stack')
def test_grids():
    return grids.GridStack.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)

@pytest.fixture(name='image')
def test_image():

    image = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
    noise_map = im.NoiseMap(array=2.0*np.ones((3,3)), pixel_scale=1.0)
    psf = im.PSF(array=3.0*np.ones((1,1)), pixel_scale=1.0)

    return im.CCD(image=image, pixel_scale=1.0, noise_map=noise_map, psf=psf, exposure_time_map=2.0 * np.ones((3, 3)),
                  background_sky_map=3.0*np.ones((3,3)))

@pytest.fixture(name='positions')
def test_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))

@pytest.fixture(name='mask')
def test_mask():
    return msk.Mask.circular(shape=((3,3)), pixel_scale=0.1, radius_arcsec=0.1)

@pytest.fixture(name='lens_image')
def test_lens_image(image, mask):
    return li.LensImage(image=image, mask=mask)

@pytest.fixture(name='fit')
def test_fit(lens_image):

    lens_galaxy = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)
    lens_subhalo = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=0.1), redshift=1.0)
    source_galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=1.0), redshift=2.0)

    tracer_normal = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                        image_plane_grid_stack=lens_image.grid_stack,
                                                        cosmology=cosmo.Planck15)
    tracer_sensitivity = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy, lens_subhalo],
                                                             source_galaxies=[source_galaxy],
                                                             image_plane_grid_stack=lens_image.grid_stack,
                                                             cosmology=cosmo.Planck15)
    return sensitivity_fit.SensitivityProfileFit(lens_image=lens_image, tracer_normal=tracer_normal,
                                                 tracer_sensitive=tracer_sensitivity)


def test__fit_sub_plot__output_dependent_on_config(fit, general_config, sensitivity_fit_plotter_path):

    sensitivity_fit_plotters.plot_fit_subplot(fit=fit, should_plot_mask=True,
                                                  output_path=sensitivity_fit_plotter_path, output_format='png')
    assert os.path.isfile(path=sensitivity_fit_plotter_path + 'sensitivity_fit.png')
    os.remove(path=sensitivity_fit_plotter_path + 'sensitivity_fit.png')