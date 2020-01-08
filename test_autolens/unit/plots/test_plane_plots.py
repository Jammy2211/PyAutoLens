import autolens as al
import pytest
import os

from os import path

from autofit import conf

directory = path.dirname(path.realpath(__file__))

@pytest.fixture(name="plane_plotter_path")
def make_plane_plotter_setup():
    return "{}/../../test_files/plotting/plane/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )

def test__all_individual_plotters__output_file_with_default_name(
    plane_7x7, sub_grid_7x7, mask_7x7, positions_7x7, plane_plotter_path, plot_patch
):

    al.plot.plane.profile_image(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        plot_in_kpc=True,
        include_critical_curves=True,
        include_caustics=True,
        array_plotter=al.plotter.array(
            output_path=plane_plotter_path, output_format="png"
        ),
    )

    assert plane_plotter_path + "profile_image.png" in plot_patch.paths

    al.plot.plane.plane_image(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        array_plotter=al.plotter.array(
            output_path=plane_plotter_path, output_format="png"
        ),
    )

    assert plane_plotter_path + "plane_image.png" in plot_patch.paths

    al.plot.plane.convergence(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        include_critical_curves=True,
        include_caustics=True,
        array_plotter=al.plotter.array(
            output_path=plane_plotter_path, output_format="png"
        ),
    )

    assert plane_plotter_path + "convergence.png" in plot_patch.paths

    al.plot.plane.potential(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        include_critical_curves=True,
        include_caustics=True,
        array_plotter=al.plotter.array(
            output_path=plane_plotter_path, output_format="png"
        ),
    )

    assert plane_plotter_path + "potential.png" in plot_patch.paths

    al.plot.plane.deflections_y(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        include_critical_curves=True,
        include_caustics=True,
        array_plotter=al.plotter.array(
            output_path=plane_plotter_path, output_format="png"
        ),
    )

    assert plane_plotter_path + "deflections_y.png" in plot_patch.paths

    al.plot.plane.deflections_x(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        include_critical_curves=True,
        include_caustics=True,
        array_plotter=al.plotter.array(
            output_path=plane_plotter_path, output_format="png"
        ),
    )

    assert plane_plotter_path + "deflections_x.png" in plot_patch.paths

    al.plot.plane.magnification(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        include_critical_curves=True,
        include_caustics=True,
        array_plotter=al.plotter.array(
            output_path=plane_plotter_path, output_format="png"
        ),
    )

    assert plane_plotter_path + "magnification.png" in plot_patch.paths

    al.plot.plane.plane_grid(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        grid_plotter=al.plotter.grid(
            output_path=plane_plotter_path, output_format="png"
        ),
    )

    assert plane_plotter_path + "plane_grid.png" in plot_patch.paths
