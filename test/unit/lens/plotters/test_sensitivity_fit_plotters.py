from autolens.lens.plotters import sensitivity_fit_plotters

from test.fixtures import *

@pytest.fixture(name='sensitivity_fit_plotter_path')
def make_sensitivity_fit_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(os.path.dirname(os.path.realpath(__file__)))


def test__fit_sub_plot__output_dependent_on_config(
        sensitivity_fit_5x5, sensitivity_fit_plotter_path, plot_patch):

    sensitivity_fit_plotters.plot_fit_subplot(
        fit=sensitivity_fit_5x5, should_plot_mask=True, extract_array_from_mask=True,
        zoom_around_mask=True, cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=sensitivity_fit_plotter_path, output_format='png')

    assert sensitivity_fit_plotter_path + 'sensitivity_fit.png' in plot_patch.paths
