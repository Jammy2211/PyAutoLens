from os import path

import pytest

import autolens.plot as aplt
from autolens.lens.plot.tracer_plots import (
    subplot_tracer,
    subplot_lensed_images,
    subplot_galaxies_images,
    fits_tracer,
    fits_source_plane_images,
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_tracer_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "tracer",
    )


def test__subplot_tracer__two_plane_tracer__output_file_created(
    tracer_x2_plane_7x7, grid_2d_7x7, plot_path, plot_patch
):
    subplot_tracer(
        tracer=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "tracer.png") in plot_patch.paths


def test__subplot_galaxies_images__two_plane_tracer__output_file_created(
    tracer_x2_plane_7x7, grid_2d_7x7, plot_path, plot_patch
):
    subplot_galaxies_images(
        tracer=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "galaxies_images.png") in plot_patch.paths


def test__subplot_lensed_images__two_plane_tracer__output_file_created(
    tracer_x2_plane_7x7, grid_2d_7x7, plot_path, plot_patch
):
    subplot_lensed_images(
        tracer=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "lensed_images.png") in plot_patch.paths


def test__fits_tracer__two_plane_tracer__tracer_fits_file_written(
    tracer_x2_plane_7x7, grid_2d_7x7, tmp_path
):
    fits_tracer(
        tracer=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        output_path=tmp_path,
    )
    assert (tmp_path / "tracer.fits").exists()


def test__fits_source_plane_images__two_plane_tracer__source_plane_fits_file_written(
    tracer_x2_plane_7x7, grid_2d_7x7, tmp_path
):
    fits_source_plane_images(
        tracer=tracer_x2_plane_7x7,
        grid=grid_2d_7x7,
        output_path=tmp_path,
    )
    assert (tmp_path / "source_plane_images.fits").exists()
