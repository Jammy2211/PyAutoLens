import itertools
import os
import shutil
from functools import wraps

import pytest
import numpy as np

from autolens import conf
from autolens.imaging import mask as msk
from autolens.plotting import profile_plotters
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp

@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../config/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path+"general.ini")

@pytest.fixture(name='profile_plotter_path')
def test_profile_plotter_setup():
    
    profile_plotter_path = "{}/../test_files/plotting/profiles/".format(os.path.dirname(os.path.realpath(__file__)))

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
    return msk.ImagingGrids.from_shape_and_pixel_scale(shape=(100, 100), pixel_scale=0.05, sub_grid_size=2)


def test__intensities_is_output(light_profile, grids, profile_plotter_path):

    profile_plotters.plot_intensities(light_profile=light_profile, grid=grids.image, 
                                      output_path=profile_plotter_path, output_format='png')
    assert os.path.isfile(path=profile_plotter_path+'intensities.png')
    os.remove(path=profile_plotter_path+'intensities.png')

def test__surface_density_is_output(mass_profile, grids, profile_plotter_path):

    profile_plotters.plot_surface_density(mass_profile=mass_profile, grid=grids.image, 
                                      output_path=profile_plotter_path, output_format='png')
    assert os.path.isfile(path=profile_plotter_path+'surface_density.png')
    os.remove(path=profile_plotter_path+'surface_density.png')
    
def test__potential_is_output(mass_profile, grids, profile_plotter_path):

    profile_plotters.plot_potential(mass_profile=mass_profile, grid=grids.image, 
                                      output_path=profile_plotter_path, output_format='png')
    assert os.path.isfile(path=profile_plotter_path+'potential.png')
    os.remove(path=profile_plotter_path+'potential.png')
    
def test__deflections_is_output(mass_profile, grids, profile_plotter_path):

    profile_plotters.plot_deflections(mass_profile=mass_profile, grid=grids.image, 
                                      output_path=profile_plotter_path, output_format='png')
    assert os.path.isfile(path=profile_plotter_path+'deflections.png')
    os.remove(path=profile_plotter_path+'deflections.png')