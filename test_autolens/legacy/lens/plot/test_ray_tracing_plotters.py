from os import path

import pytest

import autolens as al
import autolens.plot as aplt

from autolens.legacy.lens.plot.ray_tracing_plotters import TracerPlotter

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
    sub_grid_2d_7x7,
    mask_2d_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):

    galaxy = al.legacy.Galaxy(
        redshift=0.5,
        light_profile_0=al.lp.SersicSph(
            intensity=1.0, effective_radius=2.0, sersic_index=2.0
        ),
    )

    tracer = al.legacy.Tracer.from_galaxies(galaxies=[galaxy])

    tracer_plotter = TracerPlotter(
        tracer=tracer,
        grid=sub_grid_2d_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    tracer.planes[0].galaxies[0].hyper_galaxy = al.legacy.HyperGalaxy()
    tracer.planes[0].galaxies[0].adapt_model_image = al.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )
    tracer.planes[0].galaxies[0].adapt_galaxy_image = al.Array2D.ones(
        shape_native=(7, 7), pixel_scales=0.1
    )

    tracer_plotter.figures_2d(contribution_map=True)

    assert path.join(plot_path, "contribution_map_2d.png") in plot_patch.paths
