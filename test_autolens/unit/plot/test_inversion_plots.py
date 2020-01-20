from os import path

import autofit as af
import autolens.plot as aplt
import os

import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_setup():
    return "{}/../../../test_files/plotting/inversion/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    af.conf.instance = af.conf.Config(
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


def test__individual_attributes_are_output_for_rectangular_inversion(
    rectangular_inversion_7x7_3x3, positions_7x7, plot_path, plot_patch
):

    critical_curves = [(0.0, 0.0), (0.1, 0.1)]
    caustics = [(0.0, 0.0), (0.1, 0.1)]

    aplt.inversion.reconstructed_image(
        inversion=rectangular_inversion_7x7_3x3,
        image_positions=positions_7x7,
        critical_curves=critical_curves,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "reconstructed_image.png" in plot_patch.paths

    aplt.inversion.reconstruction(
        inversion=rectangular_inversion_7x7_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "reconstruction.png" in plot_patch.paths

    aplt.inversion.errors(
        inversion=rectangular_inversion_7x7_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "errors.png" in plot_patch.paths

    aplt.inversion.residual_map(
        inversion=rectangular_inversion_7x7_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "residual_map.png" in plot_patch.paths

    aplt.inversion.normalized_residual_map(
        inversion=rectangular_inversion_7x7_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "normalized_residual_map.png" in plot_patch.paths

    aplt.inversion.chi_squared_map(
        inversion=rectangular_inversion_7x7_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "chi_squared_map.png" in plot_patch.paths

    aplt.inversion.regularization_weights(
        inversion=rectangular_inversion_7x7_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "regularization_weights.png" in plot_patch.paths

    aplt.inversion.interpolated_reconstruction(
        inversion=rectangular_inversion_7x7_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "interpolated_reconstruction.png" in plot_patch.paths

    aplt.inversion.interpolated_errors(
        inversion=rectangular_inversion_7x7_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "interpolated_errors.png" in plot_patch.paths


def test__individual_attributes_are_output_for_voronoi_inversion(
    voronoi_inversion_9_3x3, positions_7x7, mask_7x7, plot_path, plot_patch
):

    critical_curves = [(0.0, 0.0), (0.1, 0.1)]
    caustics = [(0.0, 0.0), (0.1, 0.1)]

    aplt.inversion.reconstructed_image(
        inversion=voronoi_inversion_9_3x3,
        image_positions=positions_7x7,
        critical_curves=critical_curves,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "reconstructed_image.png" in plot_patch.paths

    aplt.inversion.reconstruction(
        inversion=voronoi_inversion_9_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "reconstruction.png" in plot_patch.paths

    aplt.inversion.errors(
        inversion=voronoi_inversion_9_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "errors.png" in plot_patch.paths

    aplt.inversion.residual_map(
        inversion=voronoi_inversion_9_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "residual_map.png" in plot_patch.paths

    aplt.inversion.normalized_residual_map(
        inversion=voronoi_inversion_9_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "normalized_residual_map.png" in plot_patch.paths

    aplt.inversion.chi_squared_map(
        inversion=voronoi_inversion_9_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "chi_squared_map.png" in plot_patch.paths

    aplt.inversion.regularization_weights(
        inversion=voronoi_inversion_9_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        image_pixel_indexes=[0],
        source_pixel_indexes=[1],
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "regularization_weights.png" in plot_patch.paths

    aplt.inversion.interpolated_reconstruction(
        inversion=voronoi_inversion_9_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "interpolated_reconstruction.png" in plot_patch.paths

    aplt.inversion.interpolated_errors(
        inversion=voronoi_inversion_9_3x3,
        source_positions=positions_7x7,
        caustics=caustics,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "interpolated_errors.png" in plot_patch.paths


def test__inversion_subplot_is_output_for_all_inversions(
    imaging_7x7,
    rectangular_inversion_7x7_3x3,
    voronoi_inversion_9_3x3,
    plot_path,
    plot_patch,
):
    aplt.inversion.subplot_inversion(
        inversion=rectangular_inversion_7x7_3x3,
        image_pixel_indexes=[[0, 1, 2], [3]],
        source_pixel_indexes=[[1, 2], [0]],
        sub_plotter=aplt.SubPlotter(output=aplt.Output(path=plot_path, format="png")),
    )
    assert plot_path + "subplot_inversion.png" in plot_patch.paths

    aplt.inversion.subplot_inversion(
        inversion=voronoi_inversion_9_3x3,
        image_pixel_indexes=[[0, 1, 2], [3]],
        source_pixel_indexes=[[1, 2], [0]],
        sub_plotter=aplt.SubPlotter(output=aplt.Output(path=plot_path, format="png")),
    )
    assert plot_path + "subplot_inversion.png" in plot_patch.paths


def test__inversion_individuals__output_dependent_on_input(
    rectangular_inversion_7x7_3x3, positions_7x7, plot_path, plot_patch
):

    aplt.inversion.individuals(
        inversion=rectangular_inversion_7x7_3x3,
        plot_reconstructed_image=True,
        plot_errors=True,
        plot_chi_squared_map=True,
        plot_interpolated_reconstruction=True,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "reconstructed_image.png" in plot_patch.paths
    assert plot_path + "reconstruction.png" not in plot_patch.paths
    assert plot_path + "errors.png" in plot_patch.paths
    assert plot_path + "residual_map.png" not in plot_patch.paths
    assert plot_path + "normalized_residual_map.png" not in plot_patch.paths
    assert plot_path + "chi_squared_map.png" in plot_patch.paths
    assert plot_path + "interpolated_reconstruction.png" in plot_patch.paths
    assert plot_path + "interpolated_errors.png" not in plot_patch.paths
