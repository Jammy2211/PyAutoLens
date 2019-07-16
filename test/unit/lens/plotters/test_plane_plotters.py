from autolens.lens.plotters import plane_plotters
import pytest
import os


@pytest.fixture(name="plane_plotter_path")
def make_plane_plotter_setup():
    return "{}/../../test_files/plotting/plane/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__all_individual_plotters__output_file_with_default_name(
    plane_7x7, mask_7x7, positions_7x7, plane_plotter_path, plot_patch
):

    plane_plotters.plot_image_plane_image(
        plane=plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=plane_plotter_path,
        output_format="png",
    )

    assert plane_plotter_path + "plane_image_plane_image.png" in plot_patch.paths

    plane_plotters.plot_plane_image(
        plane=plane_7x7,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=plane_plotter_path,
        output_format="png",
    )

    assert plane_plotter_path + "plane_image.png" in plot_patch.paths

    plane_plotters.plot_convergence(
        plane=plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=plane_plotter_path,
        output_format="png",
    )

    assert plane_plotter_path + "plane_convergence.png" in plot_patch.paths

    plane_plotters.plot_potential(
        plane=plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=plane_plotter_path,
        output_format="png",
    )

    assert plane_plotter_path + "plane_potential.png" in plot_patch.paths

    plane_plotters.plot_deflections_y(
        plane=plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=plane_plotter_path,
        output_format="png",
    )

    assert plane_plotter_path + "plane_deflections_y.png" in plot_patch.paths

    plane_plotters.plot_deflections_x(
        plane=plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=plane_plotter_path,
        output_format="png",
    )

    assert plane_plotter_path + "plane_deflections_x.png" in plot_patch.paths

    plane_plotters.plot_plane_grid(
        plane=plane_7x7, output_path=plane_plotter_path, output_format="png"
    )

    assert plane_plotter_path + "plane_grid.png" in plot_patch.paths
