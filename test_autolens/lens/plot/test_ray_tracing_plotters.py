from os import path

import pytest

import autolens as al
import autolens.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_tracer_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "ray_tracing",
    )


def test__all_individual_plotter(
    tracer_x2_plane_7x7,
    sub_grid_2d_7x7,
    mask_2d_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    tracer_plotter = aplt.TracerPlotter(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_2d_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    tracer_plotter.figures_2d(
        image=True,
        source_plane=True,
        convergence=True,
        potential=True,
        deflections_y=True,
        deflections_x=True,
        magnification=True,
    )

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "plane_image_of_plane_1.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths
    assert path.join(plot_path, "potential_2d.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_y_2d.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_x_2d.png") in plot_patch.paths
    assert path.join(plot_path, "magnification_2d.png") in plot_patch.paths

    tracer_x2_plane_7x7.planes[0].galaxies[0].hyper_galaxy = al.HyperGalaxy()
    tracer_x2_plane_7x7.planes[0].galaxies[0].hyper_model_image = al.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )
    tracer_x2_plane_7x7.planes[0].galaxies[0].hyper_galaxy_image = al.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )

    tracer_plotter.figures_2d(contribution_map=True)

    assert path.join(plot_path, "contribution_map_2d.png") in plot_patch.paths

    plot_patch.paths = []

    tracer_plotter = aplt.TracerPlotter(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    tracer_plotter.figures_2d(
        image=True, source_plane=True, potential=True, magnification=True
    )

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "plane_image_of_plane_1.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_2d.png") not in plot_patch.paths
    assert path.join(plot_path, "potential_2d.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_y_2d.png") not in plot_patch.paths
    assert path.join(plot_path, "deflections_x_2d.png") not in plot_patch.paths
    assert path.join(plot_path, "magnification_2d.png") in plot_patch.paths


def test__figures_of_plane(
    tracer_x2_plane_7x7,
    sub_grid_2d_7x7,
    mask_2d_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    tracer_plotter = aplt.TracerPlotter(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    tracer_plotter.figures_2d_of_planes(plane_image=True, plane_grid=True)

    assert path.join(plot_path, "plane_image_of_plane_0.png") in plot_patch.paths
    assert path.join(plot_path, "plane_image_of_plane_1.png") in plot_patch.paths

    plot_patch.paths = []

    tracer_plotter.figures_2d_of_planes(plane_image=True, plane_index=0)

    assert path.join(plot_path, "plane_image_of_plane_0.png") in plot_patch.paths
    assert path.join(plot_path, "plane_image_of_plane_1.png") not in plot_patch.paths


def test__tracer_sub_plot_output(
    tracer_x2_plane_7x7, sub_grid_2d_7x7, include_2d_all, plot_path, plot_patch
):

    tracer_plotter = aplt.TracerPlotter(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_2d_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    tracer_plotter.subplot_tracer()
    assert path.join(plot_path, "subplot_tracer.png") in plot_patch.paths

    tracer_plotter.subplot_plane_images()
    assert path.join(plot_path, "subplot_plane_images.png") in plot_patch.paths
