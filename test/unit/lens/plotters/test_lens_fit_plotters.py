from autolens.lens.plotters import lens_fit_plotters
from test.fixtures import *


@pytest.fixture(name='lens_fit_plotter_path')
def make_lens_fit_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(os.path.dirname(os.path.realpath(__file__)))

def test__fit_sub_plot_lens_only(lens_fit_x1_plane_5x5, lens_fit_plotter_path, plot_patch):
    lens_fit_plotters.plot_fit_subplot(fit=lens_fit_x1_plane_5x5, should_plot_mask=True, extract_array_from_mask=True,
                                       zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                       output_path=lens_fit_plotter_path, output_format='png')
    assert lens_fit_plotter_path + 'lens_fit.png' in plot_patch.paths


def test__fit_sub_plot_source_and_lens(lens_fit_x2_plane_5x5, lens_fit_plotter_path, plot_patch):
    lens_fit_plotters.plot_fit_subplot(fit=lens_fit_x2_plane_5x5, should_plot_mask=True, extract_array_from_mask=True,
                                       zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
                                       output_path=lens_fit_plotter_path, output_format='png')
    assert lens_fit_plotter_path + 'lens_fit.png' in plot_patch.paths


def test__fit_individuals__lens_only__depedent_on_input(lens_fit_x1_plane_5x5, lens_fit_plotter_path, plot_patch):

    lens_fit_plotters.plot_fit_individuals(
        fit=lens_fit_x1_plane_5x5,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_model_image=True,
        should_plot_chi_squared_map=True,
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'fit_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_signal_to_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_model_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_residual_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_chi_squared_map.png' in plot_patch.paths


def test__fit_individuals__source_and_lens__depedent_on_input(lens_fit_x2_plane_5x5,
                                                               lens_fit_plotter_path, plot_patch):

    lens_fit_plotters.plot_fit_individuals(
        fit=lens_fit_x2_plane_5x5,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_model_image=True,
        should_plot_lens_subtracted_image=True,
        should_plot_source_model_image=True,
        should_plot_chi_squared_map=True,
        output_path=lens_fit_plotter_path, output_format='png')

    assert lens_fit_plotter_path + 'fit_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_signal_to_noise_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_model_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_lens_subtracted_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_lens_plane_model_image.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_source_plane_model_image.png' in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_residual_map.png' not in plot_patch.paths

    assert lens_fit_plotter_path + 'fit_chi_squared_map.png' in plot_patch.paths


def test__lens_fit_for_phase__source_and_lens__depedent_on_input(lens_fit_x2_plane_5x5,
                                                               lens_fit_plotter_path, plot_patch):

    lens_fit_plotters.plot_lens_fit_for_phase(
        fit=lens_fit_x2_plane_5x5, during_analysis=False, should_plot_mask=True, positions=None,
        extract_array_from_mask=True, zoom_around_mask=True, units='arcsec',
        should_plot_image_plane_pix=True,
        should_plot_as_subplot=True,
        should_plot_all_at_end_png=False,
        should_plot_all_at_end_fits=False,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_model_image=True,
        should_plot_lens_model_image=False,
        should_plot_lens_subtracted_image=True,
        should_plot_source_model_image=True,
        should_plot_source_plane_image=False,
        should_plot_residual_map=False,
        should_plot_chi_squared_map=True,
        should_plot_regularization_weights=False,
        visualize_path=lens_fit_plotter_path)

    assert lens_fit_plotter_path + 'lens_fit.png' in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_image.png' in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_noise_map.png' not in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_signal_to_noise_map.png' not in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_model_image.png' in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_lens_subtracted_image.png' in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_lens_plane_model_image.png' not in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_source_plane_model_image.png' in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_residual_map.png' not in plot_patch.paths
    assert lens_fit_plotter_path + 'fit_chi_squared_map.png' in plot_patch.paths
