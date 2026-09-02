import logging
from pathlib import Path

import pytest

from autolens.lens.plot import critical_curves
from autolens.imaging.plot.fit_imaging_plots import (
    _compute_critical_curve_lines,
    subplot_fit,
    subplot_fit_log10,
    subplot_fit_x1_plane,
    subplot_fit_log10_x1_plane,
    subplot_of_planes,
    subplot_mappings,
    subplot_tracer_from_fit,
    subplot_fit_combined,
    subplot_fit_combined_log10,
)

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_fit_imaging_plotter_setup():
    return directory / "files" / "plots" / "fit"


def test__subplot_fit__two_plane_tracer__output_file_created(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit.png") in plot_patch.paths


def test__subplot_fit__inversion_source__mid_zoom_panel_renders(
    fit_imaging_x2_plane_inversion_7x7, plot_path, plot_patch
):
    subplot_fit(
        fit=fit_imaging_x2_plane_inversion_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit.png") in plot_patch.paths


def test__subplot_fit__single_plane_tracer__delegates_to_x1_plane_and_creates_file(
    fit_imaging_x1_plane_7x7, plot_path, plot_patch
):
    subplot_fit(
        fit=fit_imaging_x1_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_x1_plane.png") in plot_patch.paths


def test__subplot_fit_x1_plane__single_plane_tracer__output_file_created(
    fit_imaging_x1_plane_7x7, plot_path, plot_patch
):
    subplot_fit_x1_plane(
        fit=fit_imaging_x1_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_x1_plane.png") in plot_patch.paths


def test__subplot_fit_log10__two_plane_tracer__output_file_created(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit_log10(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_log10.png") in plot_patch.paths


def test__subplot_fit_log10__single_plane_tracer__delegates_to_x1_plane_and_creates_file(
    fit_imaging_x1_plane_7x7, plot_path, plot_patch
):
    subplot_fit_log10(
        fit=fit_imaging_x1_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_log10.png") in plot_patch.paths


def test__subplot_fit_log10_x1_plane__single_plane_tracer__output_file_created(
    fit_imaging_x1_plane_7x7, plot_path, plot_patch
):
    subplot_fit_log10_x1_plane(
        fit=fit_imaging_x1_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_log10.png") in plot_patch.paths


def test__subplot_of_planes__no_plane_index_specified__all_plane_files_created(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_of_planes(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "fit_of_plane_0.png") in plot_patch.paths
    assert str(plot_path / "fit_of_plane_1.png") in plot_patch.paths


def test__subplot_of_planes__plane_index_0_specified__only_plane_0_file_created(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_of_planes(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
        plane_index=0,
    )

    assert str(plot_path / "fit_of_plane_0.png") in plot_patch.paths
    assert str(plot_path / "fit_of_plane_1.png") not in plot_patch.paths


def test__subplot_tracer_from_fit__two_plane_tracer__output_file_created(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_tracer_from_fit(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "tracer.png") in plot_patch.paths


def test__subplot_fit_combined__list_of_two_fits__output_file_created(
    fit_imaging_x1_plane_7x7, fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit_combined(
        fit_list=[fit_imaging_x1_plane_7x7, fit_imaging_x2_plane_7x7],
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_combined.png") in plot_patch.paths


def test__subplot_fit_combined_log10__list_of_two_fits__output_file_created(
    fit_imaging_x1_plane_7x7, fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit_combined_log10(
        fit_list=[fit_imaging_x1_plane_7x7, fit_imaging_x2_plane_7x7],
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit_combined_log10.png") in plot_patch.paths


@pytest.mark.parametrize(
    "exc_cls",
    [ModuleNotFoundError, ValueError],
    ids=["jax_zero_contour_missing", "no_zero_crossings"],
)
def test__compute_critical_curve_lines__known_recoverable_exceptions__silent(
    monkeypatch, caplog, exc_cls
):
    """
    Two failure modes are expected and pre-handled upstream: ``jax_zero_contour``
    not installed (``ModuleNotFoundError``) and a model with no zero crossings
    (``ValueError`` raised by ``_init_guess_from_coarse_grid``). These must
    fall through silently — no WARNING log — so plot-time noise stays clean
    when the absence of critical curves is the correct rendering.
    """
    def boom(*args, **kwargs):
        raise exc_cls("synthetic failure for test")

    # _compute_critical_curve_lines lives in autolens.lens.plot.critical_curves
    # and is re-exported by fit_imaging_plots; patch/observe it at its canonical home.
    monkeypatch.setattr(critical_curves, "_critical_curves_from", boom)

    with caplog.at_level(logging.WARNING, logger=critical_curves.__name__):
        result = _compute_critical_curve_lines(tracer=None, grid=None)

    assert result == (None, None, None, None)
    assert caplog.records == [], (
        "known-recoverable failure must not emit a WARNING log"
    )


def test__compute_critical_curve_lines__unexpected_exception__logs_warning(
    monkeypatch, caplog
):
    """
    Anything OTHER than ``ModuleNotFoundError`` / ``ValueError`` is treated as
    an unexpected failure (the silent failure mode that caused the
    2026-04-19 PyAutoGalaxy zero_contour revert and the 2026-05-16 Euclid
    pipeline regression). Such failures must surface as a WARNING log with
    a traceback — never silently swallowed.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic unexpected failure for test")

    monkeypatch.setattr(critical_curves, "_critical_curves_from", boom)

    with caplog.at_level(logging.WARNING, logger=critical_curves.__name__):
        result = _compute_critical_curve_lines(tracer=None, grid=None)

    assert result == (None, None, None, None)
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None, "traceback must be attached"
    assert isinstance(record.exc_info[1], RuntimeError)


def test__subplot_mappings__parametric_source__output_file_created(
    fit_imaging_x2_plane_7x7, plot_path, plot_patch
):
    """
    The parametric (ShapeSolver) engine. The filename carries the plane index, as the
    sibling `mappings_{pixelization_index}` of the inversion subplot does.
    """
    subplot_mappings(
        fit=fit_imaging_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "mappings_1.png") in plot_patch.paths


def test__subplot_mappings__inversion_source__output_file_created(
    fit_imaging_x2_plane_inversion_7x7, plot_path, plot_patch
):
    """
    The pixelized (inversion) engine, reached through the `pix_indexes` bypass because the
    7x7 fixture reconstructs to all zeros and so has no clumps of its own.
    """
    subplot_mappings(
        fit=fit_imaging_x2_plane_inversion_7x7,
        pix_indexes=[[0, 1, 2]],
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "mappings_1.png") in plot_patch.paths


def test__subplot_mappings__unmappable_plane__raises_rather_than_drawing_nothing(
    masked_imaging_7x7, plot_path, plot_patch
):
    """
    A source plane which cannot be mapped raises, naming what the caller has to change. It
    must not draw four empty panels: a figure with no regions reads as "this source has no
    multiple images", which is a wrong answer rather than a missing one -- so
    `subplot_mappings` deliberately has no guard around the engine.

    A source plane with neither a pixelization nor a light profile is the case which
    cannot be mapped at all.
    """
    import autolens as al

    fit = al.FitImaging(
        dataset=masked_imaging_7x7,
        tracer=al.Tracer(
            galaxies=[
                al.Galaxy(
                    redshift=0.5,
                    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
                ),
                al.Galaxy(redshift=1.0),
            ]
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        subplot_mappings(fit=fit, output_path=plot_path, output_format="png")

    assert "shape=" in str(exc_info.value)
