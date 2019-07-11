from autolens.data.array import grids
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
import numpy as np
import pytest


# Arc second coordinate grid is:.

# [[[-2.0, -2.0], [-2.0, -1.0], [-2.0, 0.0], [-2.0, 1.0], [-2.0, 2.0]],
# [[[-1.0, -2.0], [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0], [-1.0, 2.0]],
# [[[ 0.0, -2.0], [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0], [ 0.0, 2.0]],
# [[[ 1.0, -2.0], [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0], [ 1.0, 2.0]],
# [[[ 2.0, -2.0], [ 2.0, -1.0], [ 2.0, 0.0], [ 2.0, 1.0], [ 2.0, 2.0]],


def test__centre_light_profile_on_grid_coordinate__peak_flux_is_correct_index():

    regular = grids.RegularGrid.from_shape_and_pixel_scale(shape=(5, 5), pixel_scale=1.0)

    sersic = lp.SphericalSersic(centre=(2.0, -2.0))
    image_1d = sersic.intensities_from_grid(grid=regular)
    image_2d = regular.array_2d_from_array_1d(array_1d=image_1d)

    assert image_1d.argmax() == 0
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (0, 0)

    sersic = lp.SphericalSersic(centre=(2.0, 2.0))
    image_1d = sersic.intensities_from_grid(grid=regular)
    image_2d = regular.array_2d_from_array_1d(array_1d=image_1d)

    assert image_1d.argmax() == 4
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (0, 4)

    sersic = lp.SphericalSersic(centre=(-2.0, -2.0))
    image_1d = sersic.intensities_from_grid(grid=regular)
    image_2d = regular.array_2d_from_array_1d(array_1d=image_1d)

    assert image_1d.argmax() == 20
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (4, 0)

    sersic = lp.SphericalSersic(centre=(-2.0, 2.0))
    image_1d = sersic.intensities_from_grid(grid=regular)
    image_2d = regular.array_2d_from_array_1d(array_1d=image_1d)

    assert image_1d.argmax() == 24
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (4, 4)

def test__centre_mass_profile_on_grid_coordinate__peak_density_is_correct_index():

    regular = grids.RegularGrid.from_shape_and_pixel_scale(shape=(5, 5), pixel_scale=1.0)

    sis = mp.SphericalIsothermal(centre=(2.0, -2.0))
    density_1d = sis.convergence_from_grid(grid=regular)
    density_2d = regular.array_2d_from_array_1d(array_1d=density_1d)

    assert density_1d.argmax() == 0
    assert np.unravel_index(density_2d.argmax(), density_2d.shape) == (0, 0)

    sis = mp.SphericalIsothermal(centre=(2.0, 2.0))
    density_1d = sis.convergence_from_grid(grid=regular)
    density_2d = regular.array_2d_from_array_1d(array_1d=density_1d)

    assert density_1d.argmax() == 4
    assert np.unravel_index(density_2d.argmax(), density_2d.shape) == (0, 4)

    sis = mp.SphericalIsothermal(centre=(-2.0, -2.0))
    density_1d = sis.convergence_from_grid(grid=regular)
    density_2d = regular.array_2d_from_array_1d(array_1d=density_1d)

    assert density_1d.argmax() == 20
    assert np.unravel_index(density_2d.argmax(), density_2d.shape) == (4, 0)

    sis =  mp.SphericalIsothermal(centre=(-2.0, 2.0))
    density_1d = sis.convergence_from_grid(grid=regular)
    density_2d = regular.array_2d_from_array_1d(array_1d=density_1d)

    assert density_1d.argmax() == 24
    assert np.unravel_index(density_2d.argmax(), density_2d.shape) == (4, 4)

def test__same_as_above__but_grid_is_to_7x7_for_simulation():

    grid_stack = grids.GridStack.grid_stack_for_simulation(shape=(5, 5), pixel_scale=1.0, psf_shape=(3, 3))

    sersic = lp.SphericalSersic(centre=(2.0, -2.0))
    image_1d = sersic.intensities_from_grid(grid=grid_stack.regular)
    assert image_1d.argmax() == 8
    image_2d = grid_stack.regular.trimmed_array_2d_from_padded_array_1d(padded_array_1d=image_1d)
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (0, 0)
    image_2d = grid_stack.regular.padded_array_2d_from_padded_array_1d(padded_array_1d=image_1d)
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (1, 1)

    sersic = lp.SphericalSersic(centre=(2.0, 2.0))
    image_1d = sersic.intensities_from_grid(grid=grid_stack.regular)
    assert image_1d.argmax() == 12
    image_2d = grid_stack.regular.trimmed_array_2d_from_padded_array_1d(padded_array_1d=image_1d)
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (0, 4)
    image_2d = grid_stack.regular.padded_array_2d_from_padded_array_1d(padded_array_1d=image_1d)
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (1, 5)

    sersic = lp.SphericalSersic(centre=(-2.0, -2.0))
    image_1d = sersic.intensities_from_grid(grid=grid_stack.regular)
    assert image_1d.argmax() == 36
    image_2d = grid_stack.regular.trimmed_array_2d_from_padded_array_1d(padded_array_1d=image_1d)
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (4, 0)
    image_2d = grid_stack.regular.padded_array_2d_from_padded_array_1d(padded_array_1d=image_1d)
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (5, 1)

    sersic = lp.SphericalSersic(centre=(-2.0, 2.0))
    image_1d = sersic.intensities_from_grid(grid=grid_stack.regular)
    assert image_1d.argmax() == 40
    image_2d = grid_stack.regular.trimmed_array_2d_from_padded_array_1d(padded_array_1d=image_1d)
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (4, 4)
    image_2d = grid_stack.regular.padded_array_2d_from_padded_array_1d(padded_array_1d=image_1d)
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (5, 5)

def test__deflection_angles():

    regular = grids.RegularGrid.from_shape_and_pixel_scale(shape=(5, 5), pixel_scale=1.0)

    sis = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
    deflections_1d = sis.deflections_from_grid(grid=regular)
    deflections_x_2d = regular.array_2d_from_array_1d(array_1d=deflections_1d[:, 1])

    assert deflections_x_2d[0,0] == pytest.approx(-1.0*deflections_x_2d[0,4], 1e-2)
    assert deflections_x_2d[1,1] == pytest.approx(-1.0*deflections_x_2d[1,3], 1e-2)
    assert deflections_x_2d[3,1] == pytest.approx(-1.0*deflections_x_2d[3,3], 1e-2)
    assert deflections_x_2d[4,0] == pytest.approx(-1.0*deflections_x_2d[4,4], 1e-2)
    assert deflections_x_2d[0,2] == pytest.approx(deflections_x_2d[4,2], 1e-2)

    sis = mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
    deflections_1d = sis.deflections_from_grid(grid=regular)
    deflections_y_2d = regular.array_2d_from_array_1d(array_1d=deflections_1d[:, 0])

    assert deflections_y_2d[0,0] == pytest.approx(-1.0*deflections_y_2d[4,0], 1e-2)
    assert deflections_y_2d[1,1] == pytest.approx(-1.0*deflections_y_2d[3,1], 1e-2)
    assert deflections_y_2d[1,3] == pytest.approx(-1.0*deflections_y_2d[3,3], 1e-2)
    assert deflections_y_2d[0,4] == pytest.approx(-1.0*deflections_y_2d[4,4], 1e-2)
    assert deflections_y_2d[2,0] == pytest.approx(deflections_y_2d[2,4], 1e-2)

# def test__move_source_galaxy_around_source_plane__peak_follows_source_direction():
#
#     regular = masks.RegularGrid.from_shape_and_pixel_scale(shape=(5, 5), pixel_scales=1.0)
#     sis = mp.SphericalIsothermal(origin=(0.0, 0.0), einstein_radius=1.0)
#     sersic = lp.SphericalSersic(origin=(1.0, 0.0))
#
#     deflections = sis.deflections_from_grid(grid=regular)
#     source_grid = np.subtract(regular, deflections)
#     print(regular[22])
#     print(deflections[22])
#     print(source_grid[22])
#     stop
#     source_image = sersic.intensities_from_grid(grid=source_grid)
#     print(source_image.argmax())
#
#     grid_stack = masks.GridStack.from_shape_and_pixel_scale(shape=(5, 5), pixel_scales=1.0)
#     lens_galaxy = g.Galaxy(mass=sis)
#     source_galaxy = g.Galaxy(light=sersic)
#     tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
#                                                  image_plane_grids=grid_stack)
#
#     print(source_grid)
#     print(tracer.source_plane.grid_stacks.regular)
#     print(np.subtract(source_grid, tracer.source_plane.grid_stacks.regular))
#     print(tracer.regular_plane_image)