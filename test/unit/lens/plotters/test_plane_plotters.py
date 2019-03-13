import numpy as np

from autolens.data.array import grids, mask as msk
from autolens.lens import plane as pl
from autolens.lens.plotters import plane_plotters
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from test.fixtures import *


@pytest.fixture(name='plane_plotter_path')
def make_plane_plotter_setup():
    return "{}/../../test_files/plotting/plane/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name='positions')
def make_positions():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))


@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask.circular(shape=((3, 3)), pixel_scale=0.1, radius_arcsec=0.1)


@pytest.fixture(name='galaxy_light')
def make_galaxy_light():
    return g.Galaxy(light=lp.EllipticalSersic(intensity=1.0))


@pytest.fixture(name='galaxy_mass')
def make_galaxy_mass():
    return g.Galaxy(mass=mp.SphericalIsothermal(einstein_radius=1.0))


@pytest.fixture(name='grid_stack')
def make_grid_stack():
    return grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=(3, 3), pixel_scale=0.05, sub_grid_size=2)


@pytest.fixture(name='plane')
def make_plane(galaxy_light, grid_stack):
    return pl.Plane(galaxies=[galaxy_light], grid_stack=grid_stack)


def test__image_plane_image_is_output(plane, mask, positions, plane_plotter_path, plot_patch):

    plane_plotters.plot_image_plane_image(plane=plane, mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                          positions=positions, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                          output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_image_plane_image.png' in plot_patch.paths


def test__plane_image_is_output(plane, positions, plane_plotter_path, plot_patch):

    plane_plotters.plot_plane_image(plane=plane, positions=positions, output_path=plane_plotter_path,
                                    cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                    output_format='png')

    assert plane_plotter_path + 'plane_image.png' in plot_patch.paths


def test__convergence_is_output(plane, mask, plane_plotter_path, plot_patch):

    plane_plotters.plot_convergence(plane=plane, mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                    cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                    output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_convergence.png' in plot_patch.paths


def test__potential_is_output(plane, mask, plane_plotter_path, plot_patch):

    plane_plotters.plot_potential(plane=plane, mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                  cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                  output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_potential.png' in plot_patch.paths


def test__deflections_y_is_output(plane, mask, plane_plotter_path, plot_patch):

    plane_plotters.plot_deflections_y(plane=plane, mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                      cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                      output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_deflections_y.png' in plot_patch.paths


def test__deflections_x_is_output(plane, mask, plane_plotter_path, plot_patch):

    plane_plotters.plot_deflections_x(plane=plane, mask=mask, extract_array_from_mask=True, zoom_around_mask=True,
                                      cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                      output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_deflections_x.png' in plot_patch.paths


def test__plane_grid_is_output(plane, plane_plotter_path, plot_patch):

    plane_plotters.plot_plane_grid(plane=plane, output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_grid.png' in plot_patch.paths
