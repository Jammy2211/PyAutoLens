from autolens.pipeline.plotters import hyper_plotters
from test.fixtures import *

import numpy as np

@pytest.fixture(name='hyper_plotter_path')
def make_hyper_plotter_setup():
    return "{}/../../test_files/plotting/hyper/".format(os.path.dirname(os.path.realpath(__file__)))

def test__plot_hyper_model_image(hyper_model_image_5x5, hyper_plotter_path, plot_patch):

    hyper_plotters.plot_hyper_model_image(
        hyper_model_image=hyper_model_image_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=hyper_plotter_path, output_format='png')

    assert hyper_plotter_path + 'hyper_model_image.png' in plot_patch.paths

def test__plot_hyper_galaxy_image(hyper_galaxy_image_0_5x5, hyper_plotter_path, plot_patch):

    hyper_plotters.plot_hyper_galaxy_image(
        hyper_galaxy_image=hyper_galaxy_image_0_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=hyper_plotter_path, output_format='png')

    assert hyper_plotter_path + 'hyper_galaxy_image.png' in plot_patch.paths

def test__plot_contribution_map(contribution_map_5x5, hyper_plotter_path, plot_patch):

    hyper_plotters.plot_contribution_map(
        contribution_map=contribution_map_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=hyper_plotter_path, output_format='png')

    assert hyper_plotter_path + 'contribution_map.png' in plot_patch.paths
    
def test__plot_hyper_noise_map(hyper_noise_map_5x5, hyper_plotter_path, plot_patch):

    hyper_plotters.plot_hyper_noise_map(
        hyper_noise_map=hyper_noise_map_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=hyper_plotter_path, output_format='png')

    assert hyper_plotter_path + 'hyper_noise_map.png' in plot_patch.paths

def test__plot_chi_squared_map(lens_fit_x1_plane_5x5, hyper_plotter_path, plot_patch):

    hyper_plotters.plot_chi_squared_map(
        chi_squared_map=lens_fit_x1_plane_5x5.chi_squared_map_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=hyper_plotter_path, output_format='png')

    assert hyper_plotter_path + 'chi_squared_map.png' in plot_patch.paths


def test__plot_hyper_chi_squared_map(lens_fit_x1_plane_5x5, hyper_plotter_path, plot_patch):

    hyper_plotters.plot_hyper_chi_squared_map(
        hyper_chi_squared_map=lens_fit_x1_plane_5x5.chi_squared_map_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=hyper_plotter_path, output_format='png')

    assert hyper_plotter_path + 'hyper_chi_squared_map.png' in plot_patch.paths


def test__plot_hyper_galaxy(
        hyper_galaxy_image_0_5x5, contribution_map_5x5, noise_map_5x5, hyper_noise_map_5x5, lens_fit_x1_plane_5x5,
        hyper_plotter_path, plot_patch):

    hyper_plotters.plot_hyper_galaxy_subplot(
        hyper_galaxy_image=hyper_galaxy_image_0_5x5,
        contribution_map=contribution_map_5x5,
        noise_map=noise_map_5x5,
        hyper_noise_map=hyper_noise_map_5x5,
        chi_squared_map=lens_fit_x1_plane_5x5.chi_squared_map_2d,
        hyper_chi_squared_map=lens_fit_x1_plane_5x5.chi_squared_map_2d,
        extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=hyper_plotter_path, output_format='png')

    assert hyper_plotter_path + 'hyper_galaxy.png' in plot_patch.paths

def test__plot_hyper_galaxy_images(
        hyper_galaxy_image_0_5x5, hyper_galaxy_image_1_5x5, mask_5x5, hyper_plotter_path, plot_patch):

    hyper_galaxy_image_path_dict = {}

    hyper_galaxy_image_path_dict[('g0',)] = hyper_galaxy_image_0_5x5
    hyper_galaxy_image_path_dict[('g1',)] = hyper_galaxy_image_1_5x5

    hyper_plotters.plot_hyper_galaxy_images_subplot(
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict, mask=mask_5x5,
        should_plot_mask=True, extract_array_from_mask=True,
        zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=hyper_plotter_path, output_format='png')

    assert hyper_plotter_path + 'hyper_galaxy_images.png' in plot_patch.paths

    hyper_plotters.plot_hyper_galaxy_cluster_images_subplot(
        hyper_galaxy_cluster_image_path_dict=hyper_galaxy_image_path_dict, mask=mask_5x5,
        should_plot_mask=True, extract_array_from_mask=True,
        zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=hyper_plotter_path, output_format='png')

    assert hyper_plotter_path + 'hyper_galaxy_cluster_images.png' in plot_patch.paths