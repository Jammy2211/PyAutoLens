import autolens as al
import autolens.plot as aplt
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
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


def test__all_individual_plotters__output_file_with_default_name(
    plane_7x7,
    sub_grid_7x7,
    mask_7x7,
    positions_7x7,
    include_all,
    plane_plotter_path,
    plot_patch,
):

    aplt.plane.profile_image(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plane_plotter_path, format="png")),
    )

    assert plane_plotter_path + "profile_image.png" in plot_patch.paths

    aplt.plane.plane_image(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plane_plotter_path, format="png")),
    )

    assert plane_plotter_path + "plane_image.png" in plot_patch.paths

    aplt.plane.convergence(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plane_plotter_path, format="png")),
    )

    assert plane_plotter_path + "convergence.png" in plot_patch.paths

    aplt.plane.potential(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plane_plotter_path, format="png")),
    )

    assert plane_plotter_path + "potential.png" in plot_patch.paths

    aplt.plane.deflections_y(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plane_plotter_path, format="png")),
    )

    assert plane_plotter_path + "deflections_y.png" in plot_patch.paths

    aplt.plane.deflections_x(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plane_plotter_path, format="png")),
    )

    assert plane_plotter_path + "deflections_x.png" in plot_patch.paths

    aplt.plane.magnification(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plane_plotter_path, format="png")),
    )

    assert plane_plotter_path + "magnification.png" in plot_patch.paths

    aplt.plane.plane_grid(
        plane=plane_7x7,
        grid=sub_grid_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plane_plotter_path, format="png")),
    )

    assert plane_plotter_path + "plane_grid.png" in plot_patch.paths

    plane_7x7.galaxies[0].hyper_galaxy = al.HyperGalaxy()
    plane_7x7.galaxies[0].hyper_model_image = al.array.ones(
        shape_2d=(7, 7), pixel_scales=0.1
    )
    plane_7x7.galaxies[0].hyper_galaxy_image = al.array.ones(
        shape_2d=(7, 7), pixel_scales=0.1
    )

    aplt.plane.contribution_map(
        plane=plane_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aplt.Plotter(output=aplt.Output(plane_plotter_path, format="png")),
    )

    assert plane_plotter_path + "contribution_map.png" in plot_patch.paths
