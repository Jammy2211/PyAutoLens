from autolens.lens.plotters import lens_fit_plotters
from test.fixtures import *


@pytest.fixture(name='lens_fit_plotter_path')
def make_lens_fit_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(os.path.dirname(os.path.realpath(__file__)))


def test__fit_sub_plot(lens_fit_x2_plane_5x5, lens_fit_plotter_path, plot_patch):

    lens_fit_plotters.plot_fit_subplot(
        fit=lens_fit_x2_plane_5x5,
        should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'lens_fit.png' in plot_patch.paths

def test__fit_for_plane_subplot(
        lens_fit_x1_plane_5x5, lens_fit_x2_plane_5x5, lens_fit_plotter_path, plot_patch):

    lens_fit_plotters.plot_fit_subplot_for_plane(
        fit=lens_fit_x1_plane_5x5, plane_index=0,
        should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'lens_fit_plane_0.png' in plot_patch.paths

    lens_fit_plotters.plot_fit_subplot_for_plane(
        fit=lens_fit_x2_plane_5x5, plane_index=0,
        should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'lens_fit_plane_0.png' in plot_patch.paths

    lens_fit_plotters.plot_fit_subplot_for_plane(
        fit=lens_fit_x2_plane_5x5, plane_index=1,
        should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'lens_fit_plane_1.png' in plot_patch.paths

def test__fit_for_planes_subplot(
        lens_fit_x1_plane_5x5, lens_fit_x2_plane_5x5, lens_fit_plotter_path, plot_patch):

    lens_fit_plotters.plot_fit_subplot_of_planes(
        fit=lens_fit_x1_plane_5x5,
        should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'lens_fit_plane_0.png' in plot_patch.paths

    lens_fit_plotters.plot_fit_subplot_of_planes(
        fit=lens_fit_x2_plane_5x5,
        should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'lens_fit_plane_0.png' in plot_patch.paths
    assert lens_fit_plotter_path + 'lens_fit_plane_1.png' in plot_patch.paths

def test__fit_individuals__source_and_lens__depedent_on_input(
        lens_fit_x1_plane_5x5, lens_fit_x2_plane_5x5, lens_fit_plotter_path, plot_patch):

    lens_fit_plotters.plot_fit_individuals(
        fit=lens_fit_x1_plane_5x5,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_model_image=True,
        should_plot_chi_squared_map=True,
        should_plot_subtracted_images_of_planes=True,
        should_plot_model_images_of_planes=True,
        should_plot_plane_images_of_planes=True,
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'fit_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_signal_to_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_model_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_residual_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_normalized_residual_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_chi_squared_map.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_subtracted_image_of_plane_0.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_model_image_of_plane_0.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_plane_image_of_plane_0.png' in plot_patch.paths

    lens_fit_plotters.plot_fit_individuals(
        fit=lens_fit_x2_plane_5x5,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_model_image=True,
        should_plot_chi_squared_map=True,
        should_plot_subtracted_images_of_planes=True,
        should_plot_model_images_of_planes=True,
        should_plot_plane_images_of_planes=True,
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'fit_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_signal_to_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_model_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_residual_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_normalized_residual_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_chi_squared_map.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_subtracted_image_of_plane_0.png' in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_subtracted_image_of_plane_1.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_model_image_of_plane_0.png' in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_model_image_of_plane_1.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_plane_image_of_plane_0.png' in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_plane_image_of_plane_1.png' in plot_patch.paths