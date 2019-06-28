from autolens.pipeline.plotters import hyper_plotters
from test.fixtures import *

@pytest.fixture(name='hyper_plotter_path')
def make_hyper_plotter_setup():
    return "{}/../../test_files/plotting/hyper/".format(os.path.dirname(os.path.realpath(__file__)))

def test__fit_sub_plot_lens_only(hyper_model_image_5x5, hyper_plotter_path, plot_patch):

    hyper_plotters.plot_hyper_model_image(
        hyper_model_image=hyper_model_image_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=hyper_plotter_path, output_format='png')

    assert hyper_plotter_path + 'hyper_model_image.png' in plot_patch.paths

def test__fit_sub_plot_source_and_lens(hyper_galaxy_image_0_5x5, hyper_galaxy_image_1_5x5, mask_5x5,
                                       hyper_plotter_path, plot_patch):

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