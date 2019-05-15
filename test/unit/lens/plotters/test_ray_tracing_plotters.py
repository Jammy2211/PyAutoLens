from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.lens import ray_tracing
from autolens.lens.plotters import ray_tracing_plotters
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.fixtures import *


@pytest.fixture(name='ray_tracing_plotter_path')
def make_ray_tracing_plotter_setup():
    return "{}/../../test_files/plotting/ray_tracing/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name='galaxy_light')
def make_galaxy_light():
    return g.Galaxy(redshift=0.5, light=lp.EllipticalSersic(intensity=1.0))


@pytest.fixture(name='galaxy_mass')
def make_galaxy_mass():
    return g.Galaxy(redshift=0.5, mass=mp.SphericalIsothermal(einstein_radius=1.0))


@pytest.fixture(name='grid_stack')
def make_grid_stack():
    return grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=(3, 3), pixel_scale=0.05, sub_grid_size=2)

@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=((3, 3)), pixel_scale=0.1, radius_arcsec=0.1)

@pytest.fixture(name='tracer')
def make_tracer(galaxy_light, galaxy_mass, grid_stack):
    return ray_tracing.TracerImageSourcePlanes(lens_galaxies=[galaxy_mass, galaxy_light],
                                               source_galaxies=[galaxy_light],
                                               image_plane_grid_stack=grid_stack)


def test__image_plane_image_is_output(tracer, mask, ray_tracing_plotter_path, plot_patch):

    ray_tracing_plotters.plot_image_plane_image(tracer=tracer, mask=mask, extract_array_from_mask=True,
                                                zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                                output_path=ray_tracing_plotter_path, output_format='png')

    assert ray_tracing_plotter_path + 'tracer_image_plane_image.png' in plot_patch.paths


def test__convergence_is_output(tracer, mask, ray_tracing_plotter_path, plot_patch):

    ray_tracing_plotters.plot_convergence(tracer=tracer, mask=mask, extract_array_from_mask=True,
                                          zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                          output_path=ray_tracing_plotter_path, output_format='png')

    assert ray_tracing_plotter_path + 'tracer_convergence.png' in plot_patch.paths


def test__potential_is_output(tracer, mask, ray_tracing_plotter_path, plot_patch):

    ray_tracing_plotters.plot_potential(tracer=tracer, mask=mask, extract_array_from_mask=True,
                                        zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                        output_path=ray_tracing_plotter_path, output_format='png')

    assert ray_tracing_plotter_path + 'tracer_potential.png' in plot_patch.paths


def test__deflections_y_is_output(tracer, mask, ray_tracing_plotter_path, plot_patch):

    ray_tracing_plotters.plot_deflections_y(tracer=tracer, mask=mask, extract_array_from_mask=True,
                                            zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                            output_path=ray_tracing_plotter_path, output_format='png')

    assert ray_tracing_plotter_path + 'tracer_deflections_y.png' in plot_patch.paths


def test__deflections_x_is_output(tracer, mask, ray_tracing_plotter_path, plot_patch):

    ray_tracing_plotters.plot_deflections_x(tracer=tracer, mask=mask, extract_array_from_mask=True,
                                            zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                            output_path=ray_tracing_plotter_path, output_format='png')

    assert ray_tracing_plotter_path + 'tracer_deflections_x.png' in plot_patch.paths


def test__tracer_sub_plot_output(tracer, mask, ray_tracing_plotter_path, plot_patch):

    ray_tracing_plotters.plot_ray_tracing_subplot(
        tracer=tracer, mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
        output_path=ray_tracing_plotter_path, output_format='png')

    assert ray_tracing_plotter_path + 'tracer.png' in plot_patch.paths


def test__tracer_individuals__dependent_on_input(tracer, mask, ray_tracing_plotter_path, plot_patch):

    ray_tracing_plotters.plot_ray_tracing_individual(
        tracer=tracer, mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
        should_plot_image_plane_image=True, should_plot_source_plane=True, should_plot_potential=True,
        output_path=ray_tracing_plotter_path, output_format='png')

    assert ray_tracing_plotter_path + 'tracer_image_plane_image.png' in plot_patch.paths

    assert ray_tracing_plotter_path + 'tracer_source_plane.png' in plot_patch.paths

    assert ray_tracing_plotter_path + 'tracer_convergence.png' not in plot_patch.paths

    assert ray_tracing_plotter_path + 'tracer_potential.png' in plot_patch.paths

    assert ray_tracing_plotter_path + 'tracer_deflections_y.png' not in plot_patch.paths

    assert ray_tracing_plotter_path + 'tracer_deflections_x.png' not in plot_patch.paths


def test__plot_ray_tracing_for_phase__dependent_on_input(tracer, mask, ray_tracing_plotter_path, plot_patch):

    ray_tracing_plotters.plot_ray_tracing_for_phase(
        tracer=tracer, during_analysis=False, mask=mask, positions=None,
        extract_array_from_mask=True, zoom_around_mask=True, units='arcsec',
        should_plot_as_subplot=True,
        should_plot_all_at_end_png=False,
        should_plot_all_at_end_fits=False,
        should_plot_image_plane_image=True,
        should_plot_source_plane=True,
        should_plot_convergence=False,
        should_plot_potential=True,
        should_plot_deflections=False,
        visualize_path=ray_tracing_plotter_path)

    assert ray_tracing_plotter_path + 'tracer.png' in plot_patch.paths
    assert ray_tracing_plotter_path + 'tracer_image_plane_image.png' in plot_patch.paths
    assert ray_tracing_plotter_path + 'tracer_source_plane.png' in plot_patch.paths
    assert ray_tracing_plotter_path + 'tracer_convergence.png' not in plot_patch.paths
    assert ray_tracing_plotter_path + 'tracer_potential.png' in plot_patch.paths
    assert ray_tracing_plotter_path + 'tracer_deflections_y.png' not in plot_patch.paths
    assert ray_tracing_plotter_path + 'tracer_deflections_x.png' not in plot_patch.paths