from autolens.lens.plotters import lens_plotter_util
from test.fixtures import *

@pytest.fixture(name='lens_plotter_util_path')
def make_lens_plotter_util_path_setup():
    return "{}/../../test_files/plotting/lens_plotter_util/".format(
        os.path.dirname(os.path.realpath(__file__)))


def test__image_is_output(
        lens_fit_x2_plane_5x5, lens_plotter_util_path, plot_patch):

    lens_plotter_util.plot_image(
        fit=lens_fit_x2_plane_5x5, 
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True, 
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_image.png' in plot_patch.paths

def test__noise_map_is_output(
        lens_fit_x2_plane_5x5, lens_plotter_util_path, plot_patch):

    lens_plotter_util.plot_noise_map(
        fit=lens_fit_x2_plane_5x5, 
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'], 
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_noise_map.png' in plot_patch.paths

def test__signal_to_noise_map_is_output(
        lens_fit_x2_plane_5x5, lens_plotter_util_path, plot_patch):
    
    lens_plotter_util.plot_signal_to_noise_map(
        fit=lens_fit_x2_plane_5x5, 
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'], 
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_signal_to_noise_map.png' in plot_patch.paths

def test__model_image_is_output(
        lens_fit_x2_plane_5x5, lens_plotter_util_path, plot_patch):

    lens_plotter_util.plot_model_data(
        fit=lens_fit_x2_plane_5x5, 
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'], 
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_model_image.png' in plot_patch.paths

def test__residual_map_is_output(
        lens_fit_x2_plane_5x5, lens_plotter_util_path, plot_patch):
    
    lens_plotter_util.plot_residual_map(
        fit=lens_fit_x2_plane_5x5, 
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'], 
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_residual_map.png' in plot_patch.paths

def test__normalized_residual_map_is_output(
        lens_fit_x2_plane_5x5, lens_plotter_util_path, plot_patch):

    lens_plotter_util.plot_normalized_residual_map(
        fit=lens_fit_x2_plane_5x5,
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_normalized_residual_map.png' in plot_patch.paths

def test__chi_squared_map_is_output(
        lens_fit_x2_plane_5x5, lens_plotter_util_path, plot_patch):

    lens_plotter_util.plot_chi_squared_map(
        fit=lens_fit_x2_plane_5x5,
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_chi_squared_map.png' in plot_patch.paths

def test__subtracted_image_of_plane_is_output(
        lens_fit_x1_plane_5x5, lens_fit_x2_plane_5x5, lens_plotter_util_path, plot_patch):

    lens_plotter_util.plot_subtracted_image_of_plane(
        fit=lens_fit_x1_plane_5x5, plane_index=0,
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_subtracted_image_of_plane_0.png' in plot_patch.paths

    lens_plotter_util.plot_subtracted_image_of_plane(
        fit=lens_fit_x2_plane_5x5, plane_index=0,
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_subtracted_image_of_plane_0.png' in plot_patch.paths

    lens_plotter_util.plot_subtracted_image_of_plane(
        fit=lens_fit_x2_plane_5x5, plane_index=1,
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_subtracted_image_of_plane_1.png' in plot_patch.paths
    
def test__model_image_of_plane_is_output(
        lens_fit_x1_plane_5x5, lens_fit_x2_plane_5x5, lens_plotter_util_path, plot_patch):

    lens_plotter_util.plot_model_image_of_plane(
        fit=lens_fit_x1_plane_5x5, plane_index=0,
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_model_image_of_plane_0.png' in plot_patch.paths

    lens_plotter_util.plot_model_image_of_plane(
        fit=lens_fit_x2_plane_5x5, plane_index=0,
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_model_image_of_plane_0.png' in plot_patch.paths

    lens_plotter_util.plot_model_image_of_plane(
        fit=lens_fit_x2_plane_5x5, plane_index=1,
        mask=lens_fit_x2_plane_5x5.mask_2d, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=lens_plotter_util_path, output_format='png')

    assert lens_plotter_util_path + 'fit_model_image_of_plane_1.png' in plot_patch.paths