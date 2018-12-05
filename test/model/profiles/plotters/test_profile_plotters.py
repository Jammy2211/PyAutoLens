import os
import shutil

import pytest

from autofit import conf
from autolens.data.array import grids
from autolens.model.profiles.plotters import profile_plotters
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp


@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path+"general.ini")

@pytest.fixture(name='profile_plotter_path')
def test_profile_plotter_setup():
    
    profile_plotter_path = "{}/../../../test_files/plotting/profiles/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(profile_plotter_path):
        shutil.rmtree(profile_plotter_path)

    os.mkdir(profile_plotter_path)

    return profile_plotter_path

@pytest.fixture(name='light_profile')
def test_light_profile():
    return lp.EllipticalSersic(intensity=1.0)

@pytest.fixture(name='mass_profile')
def test_mass_profile():
    return mp.SphericalIsothermal(einstein_radius=1.0)

@pytest.fixture(name='grids')
def test_grids():
    return grids.DataGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)


def test__intensities_is_output(light_profile, grids, profile_plotter_path):

    profile_plotters.plot_intensities(light_profile=light_profile, grid=grids.regular,
                                      output_path=profile_plotter_path, output_format='png')
    assert os.path.isfile(path=profile_plotter_path+'intensities.png')
    os.remove(path=profile_plotter_path+'intensities.png')

def test__surface_density_is_output(mass_profile, grids, profile_plotter_path):

    profile_plotters.plot_surface_density(mass_profile=mass_profile, grid=grids.regular,
                                          output_path=profile_plotter_path, output_format='png')
    assert os.path.isfile(path=profile_plotter_path+'surface_density.png')
    os.remove(path=profile_plotter_path+'surface_density.png')
    
def test__potential_is_output(mass_profile, grids, profile_plotter_path):

    profile_plotters.plot_potential(mass_profile=mass_profile, grid=grids.regular,
                                    output_path=profile_plotter_path, output_format='png')
    assert os.path.isfile(path=profile_plotter_path+'potential.png')
    os.remove(path=profile_plotter_path+'potential.png')
    
def test__deflections_y_is_output(mass_profile, grids, profile_plotter_path):

    profile_plotters.plot_deflections_y(mass_profile=mass_profile, grid=grids.regular,
                                        output_path=profile_plotter_path, output_format='png')
    assert os.path.isfile(path=profile_plotter_path+'deflections_y.png')
    os.remove(path=profile_plotter_path+'deflections_y.png')

def test__deflections_x_is_output(mass_profile, grids, profile_plotter_path):

    profile_plotters.plot_deflections_x(mass_profile=mass_profile, grid=grids.regular,
                                        output_path=profile_plotter_path, output_format='png')
    assert os.path.isfile(path=profile_plotter_path+'deflections_x.png')
    os.remove(path=profile_plotter_path+'deflections_x.png')