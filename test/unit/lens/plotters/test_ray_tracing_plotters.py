import os

import pytest

from autolens.lens.plotters import ray_tracing_plotters


@pytest.fixture(name="ray_tracing_plotter_path")
def make_ray_tracing_plotter_setup():
    return "{}/../../test_files/plotting/ray_tracing/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__all_individual_plotters(
    tracer_x2_plane_7x7, mask_7x7, ray_tracing_plotter_path, plot_patch
):
    ray_tracing_plotters.plot_image_plane_image(
        tracer=tracer_x2_plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ray_tracing_plotter_path,
        output_format="png",
    )

    assert ray_tracing_plotter_path + "tracer_image_plane_image.png" in plot_patch.paths

    ray_tracing_plotters.plot_convergence(
        tracer=tracer_x2_plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ray_tracing_plotter_path,
        output_format="png",
    )

    assert ray_tracing_plotter_path + "tracer_convergence.png" in plot_patch.paths

    ray_tracing_plotters.plot_potential(
        tracer=tracer_x2_plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ray_tracing_plotter_path,
        output_format="png",
    )

    assert ray_tracing_plotter_path + "tracer_potential.png" in plot_patch.paths

    ray_tracing_plotters.plot_deflections_y(
        tracer=tracer_x2_plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ray_tracing_plotter_path,
        output_format="png",
    )

    assert ray_tracing_plotter_path + "tracer_deflections_y.png" in plot_patch.paths

    ray_tracing_plotters.plot_deflections_x(
        tracer=tracer_x2_plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=ray_tracing_plotter_path,
        output_format="png",
    )

    assert ray_tracing_plotter_path + "tracer_deflections_x.png" in plot_patch.paths


def test__tracer_sub_plot_output(
    tracer_x2_plane_7x7, mask_7x7, ray_tracing_plotter_path, plot_patch
):
    ray_tracing_plotters.plot_ray_tracing_subplot(
        tracer=tracer_x2_plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        output_path=ray_tracing_plotter_path,
        output_format="png",
    )

    assert ray_tracing_plotter_path + "tracer.png" in plot_patch.paths


def test__tracer_individuals__dependent_on_input(
    tracer_x2_plane_7x7, mask_7x7, ray_tracing_plotter_path, plot_patch
):
    ray_tracing_plotters.plot_ray_tracing_individual(
        tracer=tracer_x2_plane_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        should_plot_image_plane_image=True,
        should_plot_source_plane=True,
        should_plot_potential=True,
        output_path=ray_tracing_plotter_path,
        output_format="png",
    )

    assert ray_tracing_plotter_path + "tracer_image_plane_image.png" in plot_patch.paths

    assert ray_tracing_plotter_path + "tracer_source_plane.png" in plot_patch.paths

    assert ray_tracing_plotter_path + "tracer_convergence.png" not in plot_patch.paths

    assert ray_tracing_plotter_path + "tracer_potential.png" in plot_patch.paths

    assert ray_tracing_plotter_path + "tracer_deflections_y.png" not in plot_patch.paths

    assert ray_tracing_plotter_path + "tracer_deflections_x.png" not in plot_patch.paths
