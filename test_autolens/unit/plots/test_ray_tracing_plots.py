import autolens as al
import os

import pytest

from os import path

from autofit import conf

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="ray_tracing_plotter_path")
def make_ray_tracing_plotter_setup():
    return "{}/../../test_files/plotting/ray_tracing/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )


def test__all_individual_plotters(
    tracer_x2_plane_7x7, sub_grid_7x7, mask_7x7, ray_tracing_plotter_path, plot_patch
):
    al.plot.tracer.profile_image(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        plotter=al.plotter.Plotter(
            output=al.plotter.Output(path=ray_tracing_plotter_path, format="png")
        ),
    )

    assert ray_tracing_plotter_path + "profile_image.png" in plot_patch.paths

    al.plot.tracer.convergence(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        plotter=al.plotter.Plotter(
            output=al.plotter.Output(path=ray_tracing_plotter_path, format="png")
        ),
    )

    assert ray_tracing_plotter_path + "convergence.png" in plot_patch.paths

    al.plot.tracer.potential(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        plotter=al.plotter.Plotter(
            output=al.plotter.Output(path=ray_tracing_plotter_path, format="png")
        ),
    )

    assert ray_tracing_plotter_path + "potential.png" in plot_patch.paths

    al.plot.tracer.deflections_y(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        plotter=al.plotter.Plotter(
            output=al.plotter.Output(path=ray_tracing_plotter_path, format="png")
        ),
    )

    assert ray_tracing_plotter_path + "deflections_y.png" in plot_patch.paths

    al.plot.tracer.deflections_x(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        plotter=al.plotter.Plotter(
            output=al.plotter.Output(path=ray_tracing_plotter_path, format="png")
        ),
    )

    assert ray_tracing_plotter_path + "deflections_x.png" in plot_patch.paths

    al.plot.tracer.magnification(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        plotter=al.plotter.Plotter(
            output=al.plotter.Output(path=ray_tracing_plotter_path, format="png")
        ),
    )

    assert ray_tracing_plotter_path + "magnification.png" in plot_patch.paths

    tracer_x2_plane_7x7.planes[0].galaxies[0].hyper_galaxy = al.HyperGalaxy()
    tracer_x2_plane_7x7.planes[0].galaxies[0].hyper_model_image = al.array.ones(shape_2d=(7,7), pixel_scales=0.1)
    tracer_x2_plane_7x7.planes[0].galaxies[0].hyper_galaxy_image = al.array.ones(shape_2d=(7,7), pixel_scales=0.1)

    al.plot.tracer.contribution_map(
        tracer=tracer_x2_plane_7x7,
        mask=mask_7x7,
        plotter=al.plotter.Plotter(
            output=al.plotter.Output(ray_tracing_plotter_path, format="png")
        ),
    )

    assert ray_tracing_plotter_path + "contribution_map.png" in plot_patch.paths

def test__tracer_sub_plot_output(
    tracer_x2_plane_7x7, sub_grid_7x7, mask_7x7, ray_tracing_plotter_path, plot_patch
):
    al.plot.tracer.subplot_tracer(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        sub_plotter=al.plotter.SubPlotter(
            output=al.plotter.Output(path=ray_tracing_plotter_path, format="png")
        ),
    )

    assert ray_tracing_plotter_path + "subplot_tracer.png" in plot_patch.paths


def test__tracer_individuals__dependent_on_input(
    tracer_x2_plane_7x7, sub_grid_7x7, mask_7x7, ray_tracing_plotter_path, plot_patch
):
    al.plot.tracer.individual(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        plot_profile_image=True,
        plot_source_plane=True,
        plot_potential=True,
        plot_magnification=True,
        plotter=al.plotter.Plotter(
            output=al.plotter.Output(path=ray_tracing_plotter_path, format="png")
        ),
    )

    assert ray_tracing_plotter_path + "profile_image.png" in plot_patch.paths

    assert ray_tracing_plotter_path + "source_plane.png" in plot_patch.paths

    assert ray_tracing_plotter_path + "convergence.png" not in plot_patch.paths

    assert ray_tracing_plotter_path + "potential.png" in plot_patch.paths

    assert ray_tracing_plotter_path + "deflections_y.png" not in plot_patch.paths

    assert ray_tracing_plotter_path + "deflections_x.png" not in plot_patch.paths

    assert ray_tracing_plotter_path + "magnification.png" in plot_patch.paths