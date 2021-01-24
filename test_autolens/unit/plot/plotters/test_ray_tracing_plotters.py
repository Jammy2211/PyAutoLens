from os import path

import pytest

import autolens as al
import autolens.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_fit_imaging_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "ray_tracing",
    )


def test__all_individual_plotter(
    tracer_x2_plane_7x7, sub_grid_7x7, mask_7x7, include_2d_all, plot_path, plot_patch
):

    tracer_plotter = aplt.TracerPlotter(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    tracer_plotter.figures(
        image=True,
        source_plane=True,
        convergence=True,
        potential=True,
        deflections_y=True,
        deflections_x=True,
        magnification=True,
    )

    print(plot_patch.paths)

    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert path.join(plot_path, "plane_image_of_plane_1.png") in plot_patch.paths
    assert path.join(plot_path, "convergence.png") in plot_patch.paths
    assert path.join(plot_path, "potential.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_y.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_x.png") in plot_patch.paths
    assert path.join(plot_path, "magnification.png") in plot_patch.paths

    tracer_x2_plane_7x7.planes[0].galaxies[0].hyper_galaxy = al.HyperGalaxy()
    tracer_x2_plane_7x7.planes[0].galaxies[0].hyper_model_image = al.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )
    tracer_x2_plane_7x7.planes[0].galaxies[0].hyper_galaxy_image = al.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )

    tracer_plotter.figures(contribution_map=True)

    assert path.join(plot_path, "contribution_map.png") in plot_patch.paths

    plot_patch.paths = []

    tracer_plotter = aplt.TracerPlotter(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    tracer_plotter.figures(
        image=True, source_plane=True, potential=True, magnification=True
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert path.join(plot_path, "plane_image_of_plane_1.png") in plot_patch.paths
    assert path.join(plot_path, "convergence.png") not in plot_patch.paths
    assert path.join(plot_path, "potential.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_y.png") not in plot_patch.paths
    assert path.join(plot_path, "deflections_x.png") not in plot_patch.paths
    assert path.join(plot_path, "magnification.png") in plot_patch.paths


def test__tracer_sub_plot_output(
    tracer_x2_plane_7x7, sub_grid_7x7, include_2d_all, plot_path, plot_patch
):
    tracer_plotter = aplt.TracerPlotter(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    tracer_plotter.subplot_tracer()
    assert path.join(plot_path, "subplot_tracer.png") in plot_patch.paths
