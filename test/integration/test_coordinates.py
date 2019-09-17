import autolens as al
import numpy as np
import pytest


# Arc second coordinate grid is:.

# [[[-2.0, -2.0], [-2.0, -1.0], [-2.0, 0.0], [-2.0, 1.0], [-2.0, 2.0]],
# [[[-1.0, -2.0], [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0], [-1.0, 2.0]],
# [[[ 0.0, -2.0], [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0], [ 0.0, 2.0]],
# [[[ 1.0, -2.0], [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0], [ 1.0, 2.0]],
# [[[ 2.0, -2.0], [ 2.0, -1.0], [ 2.0, 0.0], [ 2.0, 1.0], [ 2.0, 2.0]],


def test__centre_light_profile_on_grid_coordinate__peak_flux_is_correct_index():

    grid = al.Grid.from_shape_pixel_scale_and_sub_size(
        shape=(5, 5), pixel_scale=1.0, sub_size=1
    )

    sersic = al.light_profiles.SphericalSersic(centre=(2.0, -2.0))
    image_1d = sersic.profile_image_from_grid(grid=grid, return_in_2d=False)
    image_2d = sersic.profile_image_from_grid(grid=grid, return_in_2d=True)

    assert image_1d.argmax() == 0
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (0, 0)

    sersic = al.light_profiles.SphericalSersic(centre=(2.0, 2.0))
    image_1d = sersic.profile_image_from_grid(grid=grid, return_in_2d=False)
    image_2d = sersic.profile_image_from_grid(grid=grid, return_in_2d=True)

    assert image_1d.argmax() == 4
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (0, 4)

    sersic = al.light_profiles.SphericalSersic(centre=(-2.0, -2.0))
    image_1d = sersic.profile_image_from_grid(grid=grid, return_in_2d=False)
    image_2d = sersic.profile_image_from_grid(grid=grid, return_in_2d=True)

    assert image_1d.argmax() == 20
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (4, 0)

    sersic = al.light_profiles.SphericalSersic(centre=(-2.0, 2.0))
    image_1d = sersic.profile_image_from_grid(grid=grid, return_in_2d=False)
    image_2d = sersic.profile_image_from_grid(grid=grid, return_in_2d=True)

    assert image_1d.argmax() == 24
    assert np.unravel_index(image_2d.argmax(), image_2d.shape) == (4, 4)


def test__centre_mass_profile_on_grid_coordinate__peak_density_is_correct_index():

    grid = al.Grid.from_shape_pixel_scale_and_sub_size(
        shape=(5, 5), pixel_scale=1.0, sub_size=1
    )

    sis = al.mass_profiles.SphericalIsothermal(centre=(2.0, -2.0))
    density_1d = sis.convergence_from_grid(grid=grid, return_in_2d=False)
    density_2d = sis.convergence_from_grid(grid=grid, return_in_2d=True)

    assert density_1d.argmax() == 0
    assert np.unravel_index(density_2d.argmax(), density_2d.shape) == (0, 0)

    sis = al.mass_profiles.SphericalIsothermal(centre=(2.0, 2.0))
    density_1d = sis.convergence_from_grid(grid=grid, return_in_2d=False)
    density_2d = sis.convergence_from_grid(grid=grid, return_in_2d=True)

    assert density_1d.argmax() == 4
    assert np.unravel_index(density_2d.argmax(), density_2d.shape) == (0, 4)

    sis = al.mass_profiles.SphericalIsothermal(centre=(-2.0, -2.0))
    density_1d = sis.convergence_from_grid(grid=grid, return_in_2d=False)
    density_2d = sis.convergence_from_grid(grid=grid, return_in_2d=True)

    assert density_1d.argmax() == 20
    assert np.unravel_index(density_2d.argmax(), density_2d.shape) == (4, 0)

    sis = al.mass_profiles.SphericalIsothermal(centre=(-2.0, 2.0))
    density_1d = sis.convergence_from_grid(grid=grid, return_in_2d=False)
    density_2d = sis.convergence_from_grid(grid=grid, return_in_2d=True)

    assert density_1d.argmax() == 24
    assert np.unravel_index(density_2d.argmax(), density_2d.shape) == (4, 4)


def test__deflection_angles():

    grid = al.Grid.from_shape_pixel_scale_and_sub_size(
        shape=(5, 5), pixel_scale=1.0, sub_size=1
    )

    sis = al.mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
    deflections_1d = sis.deflections_from_grid(grid=grid, return_in_2d=False)
    deflections_y_2d = sis.deflections_from_grid(grid=grid, return_in_2d=True)[:, :, 0]

    assert deflections_y_2d[0, 0] == pytest.approx(-1.0 * deflections_y_2d[4, 0], 1e-2)
    assert deflections_y_2d[1, 1] == pytest.approx(-1.0 * deflections_y_2d[3, 1], 1e-2)
    assert deflections_y_2d[1, 3] == pytest.approx(-1.0 * deflections_y_2d[3, 3], 1e-2)
    assert deflections_y_2d[0, 4] == pytest.approx(-1.0 * deflections_y_2d[4, 4], 1e-2)
    assert deflections_y_2d[2, 0] == pytest.approx(deflections_y_2d[2, 4], 1e-2)

    sis = al.mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
    deflections_1d = sis.deflections_from_grid(grid=grid, return_in_2d=False)
    deflections_x_2d = sis.deflections_from_grid(grid=grid, return_in_2d=True)[:, :, 1]

    assert deflections_x_2d[0, 0] == pytest.approx(-1.0 * deflections_x_2d[0, 4], 1e-2)
    assert deflections_x_2d[1, 1] == pytest.approx(-1.0 * deflections_x_2d[1, 3], 1e-2)
    assert deflections_x_2d[3, 1] == pytest.approx(-1.0 * deflections_x_2d[3, 3], 1e-2)
    assert deflections_x_2d[4, 0] == pytest.approx(-1.0 * deflections_x_2d[4, 4], 1e-2)
    assert deflections_x_2d[0, 2] == pytest.approx(deflections_x_2d[4, 2], 1e-2)


# def test__move_source_galaxy_around_source_plane__peak_follows_source_direction():
#
#     = masks.Grid.from_shape_pixel_scale_and_sub_size(shape=(5, 5), pixel_scales=1.0)
#     sis = al.mass_profiles.SphericalIsothermal(origin=(0.0, 0.0), einstein_radius=1.0)
#     sersic = al.light_profiles.SphericalSersic(origin=(1.0, 0.0))
#
#     deflections = sis.deflections_from_grid(grid=grid)
#     source_grid = np.subtract(grid, deflections)
#     print(grid[22])
#     print(deflections[22])
#     print(source_grid[22])
#     stop
#     source_image = sersic.image_from_grid(grid=source_grid)
#     print(source_image.argmax())
#
#     grid = masks.GridStack.from_shape_and_pixel_scale(shape=(5, 5), pixel_scales=1.0)
#     lens_galaxy = al.Galaxy(mass=sis)
#     source_galaxy = al.Galaxy(light=sersic)
#     tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy], galaxies=[source_galaxy],
#                                                  image_plane_grids=grid)
#
#     print(source_grid)
#     print(tracer.source_plane.grid_stacks.grid)
#     print(np.subtract(source_grid, tracer.source_plane.grid_stacks.grid))
#     print(tracer.regular_plane_image)
