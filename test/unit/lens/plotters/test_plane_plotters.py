from autolens.lens.plotters import plane_plotters
from test.fixtures import *


@pytest.fixture(name='plane_plotter_path')
def make_plane_plotter_setup():
    return "{}/../../test_files/plotting/plane/".format(os.path.dirname(os.path.realpath(__file__)))


def test__all_individual_plotters__output_file_with_default_name(
        plane_5x5, mask_5x5, positions_5x5, plane_plotter_path, plot_patch):

    plane_plotters.plot_image_plane_image(
        plane=plane_5x5, mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        positions=positions_5x5,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_image_plane_image.png' in plot_patch.paths

    plane_plotters.plot_plane_image(
        plane=plane_5x5, positions=positions_5x5,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_image.png' in plot_patch.paths

    plane_plotters.plot_convergence(
        plane=plane_5x5, mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_convergence.png' in plot_patch.paths

    plane_plotters.plot_potential(
        plane=plane_5x5, mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_potential.png' in plot_patch.paths

    plane_plotters.plot_deflections_y(
        plane=plane_5x5, mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_deflections_y.png' in plot_patch.paths

    plane_plotters.plot_deflections_x(
        plane=plane_5x5, mask=mask_5x5, extract_array_from_mask=True, zoom_around_mask=True,
        cb_tick_values=[1.0], cb_tick_labels=['1.0'],
        output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_deflections_x.png' in plot_patch.paths

    plane_plotters.plot_plane_grid(
        plane=plane_5x5, output_path=plane_plotter_path, output_format='png')

    assert plane_plotter_path + 'plane_grid.png' in plot_patch.paths
