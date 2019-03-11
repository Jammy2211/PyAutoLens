import numpy as np
from astropy import cosmology as cosmo

from autolens.data import ccd as im
from autolens.data.array import grids, mask as msk, scaled_array
from autolens.lens import lens_data as li, lens_fit
from autolens.lens import ray_tracing
from autolens.lens.plotters import lens_fit_plotters
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.fixtures import *


@pytest.fixture(name='lens_fit_plotter_path')
def make_lens_fit_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name='galaxy_light')
def make_galaxy_light():
    return g.Galaxy(light=lp.EllipticalSersic(intensity=1.0), redshift=2.0)


@pytest.fixture(name='galaxy_mass')
def make_galaxy_mass():
    return g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0), redshift=1.0)


@pytest.fixture(name='grid_stack')
def make_grid_stack():
    return grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)


@pytest.fixture(name='image')
def make_image():
    image = scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=1.0)
    noise_map = im.NoiseMap(array=2.0 * np.ones((3, 3)), pixel_scale=1.0)
    psf = im.PSF(array=3.0 * np.ones((1, 1)), pixel_scale=1.0)

    return im.CCDData(image=image, pixel_scale=1.0, noise_map=noise_map, psf=psf)


@pytest.fixture(name='positions')
def make_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=((3, 3)), pixel_scale=0.1, radius_arcsec=0.1)


@pytest.fixture(name='lens_data')
def make_lens_image(image, mask):
    return li.LensData(ccd_data=image, mask=mask)


@pytest.fixture(name='fit_lens_only')
def make_fit_lens_only(lens_data, galaxy_light):
    tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grid_stack=lens_data.grid_stack,
                                          cosmology=cosmo.Planck15)
    return lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)


@pytest.fixture(name='fit_source_and_lens')
def make_fit_source_and_lens(lens_data, galaxy_light, galaxy_mass):
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass], source_galaxies=[galaxy_light],
                                                 image_plane_grid_stack=lens_data.grid_stack, cosmology=cosmo.Planck15)
    return lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)


def test__fit_sub_plot_lens_only(fit_lens_only, lens_fit_plotter_path, plot_patch):
    lens_fit_plotters.plot_fit_subplot(fit=fit_lens_only, should_plot_mask=True, extract_array_from_mask=True,
                                       zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                       output_path=lens_fit_plotter_path, output_format='png')
    assert lens_fit_plotter_path + 'lens_fit.png' in plot_patch.paths


def test__fit_sub_plot_source_and_lens(fit_source_and_lens, lens_fit_plotter_path, plot_patch):
    lens_fit_plotters.plot_fit_subplot(fit=fit_source_and_lens, should_plot_mask=True, extract_array_from_mask=True,
                                       zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                       output_path=lens_fit_plotter_path, output_format='png')
    assert lens_fit_plotter_path + 'lens_fit.png' in plot_patch.paths


def test__fit_individuals__lens_only__depedent_on_input(fit_lens_only, lens_fit_plotter_path, plot_patch):

    lens_fit_plotters.plot_fit_individuals(
        fit=fit_lens_only,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_model_image=True,
        should_plot_chi_squared_map=True,
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'fit_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_signal_to_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_model_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_residual_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_chi_squared_map.png' in plot_patch.paths


def test__fit_individuals__source_and_lens__depedent_on_input(fit_source_and_lens,
                                                               lens_fit_plotter_path, plot_patch):

    lens_fit_plotters.plot_fit_individuals(
        fit=fit_source_and_lens,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_model_image=True,
        should_plot_lens_subtracted_image=True,
        should_plot_source_model_image=True,
        should_plot_chi_squared_map=True,
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'fit_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_signal_to_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_model_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_lens_subtracted_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_lens_plane_model_image.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_source_plane_model_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_residual_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_chi_squared_map.png' in plot_patch.paths
