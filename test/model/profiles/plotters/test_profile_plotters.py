from autolens.data.array import grids
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.model.profiles.plotters import profile_plotters
from test.fixtures import *


@pytest.fixture(name='profile_plotter_path')
def make_profile_plotter_setup():
    return "{}/../../../test_files/plotting/profiles/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name='light_profile')
def make_light_profile():
    return lp.EllipticalSersic(intensity=1.0)


@pytest.fixture(name='mass_profile')
def make_mass_profile():
    return mp.SphericalIsothermal(einstein_radius=1.0)


@pytest.fixture(name='grid_stack')
def make_grid_stack():
    return grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)


def test__intensities_is_output(light_profile, grid_stack, profile_plotter_path, plot_patch):
    profile_plotters.plot_intensities(light_profile=light_profile, grid=grid_stack.regular,
                                      output_path=profile_plotter_path, output_format='png')
    assert profile_plotter_path + 'intensities.png' in plot_patch.paths


def test__surface_density_is_output(mass_profile, grid_stack, profile_plotter_path, plot_patch):
    profile_plotters.plot_surface_density(mass_profile=mass_profile, grid=grid_stack.regular,
                                          output_path=profile_plotter_path, output_format='png')
    assert profile_plotter_path + 'surface_density.png' in plot_patch.paths


def test__potential_is_output(mass_profile, grid_stack, profile_plotter_path, plot_patch):
    profile_plotters.plot_potential(mass_profile=mass_profile, grid=grid_stack.regular,
                                    output_path=profile_plotter_path, output_format='png')
    assert profile_plotter_path + 'potential.png' in plot_patch.paths


def test__deflections_y_is_output(mass_profile, grid_stack, profile_plotter_path, plot_patch):
    profile_plotters.plot_deflections_y(mass_profile=mass_profile, grid=grid_stack.regular,
                                        output_path=profile_plotter_path, output_format='png')
    assert profile_plotter_path + 'deflections_y.png' in plot_patch.paths


def test__deflections_x_is_output(mass_profile, grid_stack, profile_plotter_path, plot_patch):
    profile_plotters.plot_deflections_x(mass_profile=mass_profile, grid=grid_stack.regular,
                                        output_path=profile_plotter_path, output_format='png')
    assert profile_plotter_path + 'deflections_x.png' in plot_patch.paths
