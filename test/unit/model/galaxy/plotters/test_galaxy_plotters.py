import autolens as al
import os

import pytest


@pytest.fixture(name="galaxy_plotter_path")
def make_galaxy_plotter_setup():
    return "{}/../../../test_files/plotting/model_galaxy/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__individual_images_are_output(
    gal_x1_lp_x1_mp,
    sub_grid_7x7,
    mask_7x7,
    positions_7x7,
    galaxy_plotter_path,
    plot_patch,
):
    al.galaxy_plotters.plot_profile_image(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        plot_critical_curves=True,
        plot_caustics=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=galaxy_plotter_path,
        output_format="png",
    )

    assert galaxy_plotter_path + "galaxy_profile_image.png" in plot_patch.paths

    al.galaxy_plotters.plot_convergence(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        plot_critical_curves=False,
        plot_caustics=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=galaxy_plotter_path,
        output_format="png",
    )

    assert galaxy_plotter_path + "galaxy_convergence.png" in plot_patch.paths

    al.galaxy_plotters.plot_potential(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        plot_critical_curves=False,
        plot_caustics=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=galaxy_plotter_path,
        output_format="png",
    )

    assert galaxy_plotter_path + "galaxy_potential.png" in plot_patch.paths

    al.galaxy_plotters.plot_deflections_y(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        plot_critical_curves=False,
        plot_caustics=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=galaxy_plotter_path,
        output_format="png",
    )

    assert galaxy_plotter_path + "galaxy_deflections_y.png" in plot_patch.paths

    al.galaxy_plotters.plot_deflections_x(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        plot_critical_curves=False,
        plot_caustics=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=galaxy_plotter_path,
        output_format="png",
    )

    assert galaxy_plotter_path + "galaxy_deflections_x.png" in plot_patch.paths

    al.galaxy_plotters.plot_magnification(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        plot_critical_curves=False,
        plot_caustics=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=galaxy_plotter_path,
        output_format="png",
    )

    assert galaxy_plotter_path + "galaxy_magnification.png" in plot_patch.paths


def test__individual_galaxy_quantities__all_are_output(
    gal_x1_lp_x1_mp,
    sub_grid_7x7,
    mask_7x7,
    positions_7x7,
    galaxy_plotter_path,
    plot_patch,
):
    al.galaxy_plotters.plot_image_subplot(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=galaxy_plotter_path,
        output_format="png",
    )

    assert galaxy_plotter_path + "galaxy_individual_image.png" in plot_patch.paths

    al.galaxy_plotters.plot_convergence_subplot(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=galaxy_plotter_path,
        output_format="png",
    )

    assert galaxy_plotter_path + "galaxy_individual_convergence.png" in plot_patch.paths

    al.galaxy_plotters.plot_potential_subplot(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=galaxy_plotter_path,
        output_format="png",
    )

    assert galaxy_plotter_path + "galaxy_individual_potential.png" in plot_patch.paths

    al.galaxy_plotters.plot_deflections_y_subplot(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=galaxy_plotter_path,
        output_format="png",
    )

    assert (
        galaxy_plotter_path + "galaxy_individual_deflections_y.png" in plot_patch.paths
    )

    al.galaxy_plotters.plot_image_subplot(
        galaxy=gal_x1_lp_x1_mp,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=galaxy_plotter_path,
        output_format="png",
    )

    assert galaxy_plotter_path + "galaxy_individual_image.png" in plot_patch.paths
