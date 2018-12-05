import os
import shutil
import pytest
import numpy as np

from autofit import conf
from autolens.data.array import scaled_array
from autolens.data.array import mask as msk
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp
from autolens.model.galaxy import galaxy as g, galaxy_fitting, galaxy_data as gd
from autolens.model.galaxy.plotters import galaxy_fitting_plotters


@pytest.fixture(name='general_config')
def test_general_config():
    general_config_path = "{}/../../../test_files/configs/plotting/".format(os.path.dirname(os.path.realpath(__file__)))
    conf.instance.general = conf.NamedConfig(general_config_path + "general.ini")


@pytest.fixture(name='galaxy_fitting_plotter_path')
def test_galaxy_fitting_plotter_setup():
    galaxy_plotter_path = "{}/../../../test_files/plotting/galaxy_fitting/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(galaxy_plotter_path):
        shutil.rmtree(galaxy_plotter_path)

    os.mkdir(galaxy_plotter_path)

    return galaxy_plotter_path

@pytest.fixture(name='galaxy')
def test_galaxy():
    return g.Galaxy(light=lp.SphericalSersic(intensity=1.0), mass=mp.SphericalIsothermal(einstein_radius=1.0))

@pytest.fixture(name='array')
def test_array():
    return scaled_array.ScaledSquarePixelArray(array=np.ones((3, 3)), pixel_scale=3.0)

@pytest.fixture(name='mask')
def test_mask():
    return msk.Mask.circular(shape=((3,3)), pixel_scale=0.1, radius_arcsec=0.1)

@pytest.fixture(name='galaxy_data_intensities')
def test_galaxy_data_intensities(array, mask):
    return gd.GalaxyDataIntensities(array=array, noise_map=np.ones((3,3)), mask=mask, sub_grid_size=2)

@pytest.fixture(name='galaxy_data_surface_density')
def test_galaxy_data_surface_density(array, mask):
    return gd.GalaxyDataSurfaceDensity(array=array, noise_map=np.ones((3,3)), mask=mask, sub_grid_size=2)

@pytest.fixture(name='galaxy_data_potential')
def test_galaxy_data_potential(array, mask):
    return gd.GalaxyDataPotential(array=array, noise_map=np.ones((3,3)), mask=mask, sub_grid_size=2)

@pytest.fixture(name='galaxy_data_deflections_y')
def test_galaxy_data_deflections_y(array, mask):
    return gd.GalaxyDataDeflectionsY(array=array, noise_map=np.ones((3,3)), mask=mask, sub_grid_size=2)

@pytest.fixture(name='galaxy_data_deflections_x')
def test_galaxy_data_deflections_x(array, mask):
    return gd.GalaxyDataDeflectionsX(array=array, noise_map=np.ones((3,3)), mask=mask, sub_grid_size=2)

@pytest.fixture(name='fit_intensities')
def test_galaxy_fitting_intensities(galaxy_data_intensities, galaxy):
    return galaxy_fitting.GalaxyFit(galaxy_datas=[galaxy_data_intensities], model_galaxy=galaxy)

@pytest.fixture(name='fit_surface_density')
def test_galaxy_fitting_surface_density(galaxy_data_surface_density, galaxy):
    return galaxy_fitting.GalaxyFit(galaxy_datas=[galaxy_data_surface_density], model_galaxy=galaxy)

@pytest.fixture(name='fit_potential')
def test_galaxy_fitting_potential(galaxy_data_potential, galaxy):
    return galaxy_fitting.GalaxyFit(galaxy_datas=[galaxy_data_potential], model_galaxy=galaxy)

@pytest.fixture(name='fit_deflections')
def test_galaxy_fitting_deflections(galaxy_data_deflections_y, galaxy_data_deflections_x, galaxy):
    return galaxy_fitting.GalaxyFitDeflections(galaxy_datas=[galaxy_data_deflections_y, galaxy_data_deflections_x],
                                               model_galaxy=galaxy)

def test__fit_sub_plot__galaxy_intensities__output_dependent_on_config(fit_intensities, general_config,
                                                                           galaxy_fitting_plotter_path):

    galaxy_fitting_plotters.plot_single_subplot(fit=fit_intensities, should_plot_mask=True,
                                                output_path=galaxy_fitting_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_fitting_plotter_path + 'galaxy_fit.png')
    os.remove(path=galaxy_fitting_plotter_path + 'galaxy_fit.png')

def test__fit_sub_plot__galaxy_surface_density__output_dependent_on_config(fit_surface_density, general_config,
                                                                           galaxy_fitting_plotter_path):

    galaxy_fitting_plotters.plot_single_subplot(fit=fit_surface_density, should_plot_mask=True,
                                                output_path=galaxy_fitting_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_fitting_plotter_path + 'galaxy_fit.png')
    os.remove(path=galaxy_fitting_plotter_path + 'galaxy_fit.png')
    
def test__fit_sub_plot__galaxy_potential__output_dependent_on_config(fit_potential, general_config,
                                                                           galaxy_fitting_plotter_path):

    galaxy_fitting_plotters.plot_single_subplot(fit=fit_potential, should_plot_mask=True,
                                                output_path=galaxy_fitting_plotter_path, output_format='png')
    assert os.path.isfile(path=galaxy_fitting_plotter_path + 'galaxy_fit.png')
    os.remove(path=galaxy_fitting_plotter_path + 'galaxy_fit.png')

def test__fit_sub_plot__galaxy_deflections__output_dependent_on_config(fit_deflections,
                                                                       general_config, galaxy_fitting_plotter_path):

    galaxy_fitting_plotters.plot_deflections_subplot(fit=fit_deflections,
                                                     should_plot_mask=True, output_path=galaxy_fitting_plotter_path,
                                                     output_format='png')
    assert os.path.isfile(path=galaxy_fitting_plotter_path + 'galaxy_fit.png')
    os.remove(path=galaxy_fitting_plotter_path + 'galaxy_fit.png')