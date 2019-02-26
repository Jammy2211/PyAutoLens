import numpy as np
from astropy import cosmology as cosmo

from autolens.data import ccd as im
from autolens.data.array import grids, mask as msk, scaled_array
from autolens.lens import lens_data as li
from autolens.lens import ray_tracing
from autolens.lens import sensitivity_fit
from autolens.lens.plotters import sensitivity_fit_plotters
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.fixtures import *


@pytest.fixture(name='sensitivity_fit_plotter_path')
def make_sensitivity_fit_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name='grid_stack')
def make_grid_stack():
    return grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=(3, 3), pixel_scale=0.05, sub_grid_size=2)


@pytest.fixture(name='ccd')
def make_ccd():
    image = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
    noise_map = im.NoiseMap(array=2.0 * np.ones((3, 3)), pixel_scale=1.0)
    psf = im.PSF(array=3.0 * np.ones((1, 1)), pixel_scale=1.0)

    return im.CCDData(image=image, pixel_scale=1.0, noise_map=noise_map, psf=psf,
                      exposure_time_map=2.0 * np.ones((3, 3)),
                      background_sky_map=3.0 * np.ones((3, 3)))


@pytest.fixture(name='positions')
def make_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=((3, 3)), pixel_scale=0.1, radius_arcsec=0.1)


@pytest.fixture(name='lens_data')
def make_lens_image(ccd, mask):
    return li.LensData(ccd_data=ccd, mask=mask)


@pytest.fixture(name='fit')
def make_fit(lens_data):
    lens_galaxy = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)
    lens_subhalo = g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=0.1), redshift=1.0)
    source_galaxy = g.Galaxy(light=lp.EllipticalSersic(intensity=1.0), redshift=2.0)

    tracer_normal = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                        image_plane_grid_stack=lens_data.grid_stack,
                                                        cosmology=cosmo.Planck15)
    tracer_sensitivity = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy, lens_subhalo],
                                                             source_galaxies=[source_galaxy],
                                                             image_plane_grid_stack=lens_data.grid_stack,
                                                             cosmology=cosmo.Planck15)
    return sensitivity_fit.SensitivityProfileFit(lens_data=lens_data, tracer_normal=tracer_normal,
                                                 tracer_sensitive=tracer_sensitivity)


def test__fit_sub_plot__output_dependent_on_config(fit, sensitivity_fit_plotter_path, plot_patch):

    sensitivity_fit_plotters.plot_fit_subplot(fit=fit, should_plot_mask=True, extract_array_from_mask=True,
                                              zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                              output_path=sensitivity_fit_plotter_path, output_format='png')

    assert sensitivity_fit_plotter_path + 'sensitivity_fit.png' in plot_patch.paths
