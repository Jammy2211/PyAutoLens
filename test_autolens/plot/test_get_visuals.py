from os import path
import pytest

import autolens.plot as aplt

from autolens.plot.get_visuals import GetVisuals2D

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_profile_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )


def test__2d__via_tracer(tracer_x2_plane_7x7, grid_2d_7x7):

    visuals_2d = aplt.Visuals2D(vectors=2)

    include_2d = aplt.Include2D(
        origin=True,
        border=True,
        light_profile_centres=True,
        mass_profile_centres=True,
        critical_curves=True,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_tracer_from(
        tracer=tracer_x2_plane_7x7, grid=grid_2d_7x7, plane_index=0
    )

    assert visuals_2d_via.origin.in_list == [(0.0, 0.0)]
    assert (visuals_2d_via.border == grid_2d_7x7.mask.border_grid_sub_1.binned).all()
    assert visuals_2d_via.light_profile_centres.in_list == [
        tracer_x2_plane_7x7.galaxies[1].light_profile_0.centre
    ]
    assert visuals_2d_via.mass_profile_centres.in_list == [
        tracer_x2_plane_7x7.galaxies[0].mass_profile_0.centre
    ]
    assert (
        visuals_2d_via.critical_curves[0]
        == tracer_x2_plane_7x7.critical_curves_from(grid=grid_2d_7x7)[0]
    ).all()
    assert visuals_2d_via.vectors == 2

    include_2d = aplt.Include2D(
        origin=True,
        border=True,
        light_profile_centres=True,
        mass_profile_centres=True,
        caustics=True,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_tracer_from(
        tracer=tracer_x2_plane_7x7, grid=grid_2d_7x7, plane_index=1
    )

    assert visuals_2d_via.origin.in_list == [(0.0, 0.0)]
    traced_border = tracer_x2_plane_7x7.traced_grid_list_from(
        grid=grid_2d_7x7.mask.border_grid_sub_1.binned
    )[1]
    assert (visuals_2d_via.border == traced_border).all()
    assert visuals_2d_via.light_profile_centres.in_list == [
        tracer_x2_plane_7x7.galaxies[1].light_profile_0.centre
    ]
    assert visuals_2d_via.mass_profile_centres is None
    assert (
        visuals_2d_via.caustics[0]
        == tracer_x2_plane_7x7.caustics_from(grid=grid_2d_7x7)[0]
    ).all()

    include_2d = aplt.Include2D(
        origin=False,
        border=False,
        light_profile_centres=False,
        mass_profile_centres=False,
        critical_curves=False,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_tracer_from(
        tracer=tracer_x2_plane_7x7, grid=grid_2d_7x7, plane_index=0
    )

    assert visuals_2d_via.origin is None
    assert visuals_2d_via.border is None
    assert visuals_2d_via.light_profile_centres is None
    assert visuals_2d_via.mass_profile_centres is None
    assert visuals_2d_via.critical_curves is None
    assert visuals_2d_via.vectors == 2


def test__via_fit_imaging_from(fit_imaging_x2_plane_7x7, grid_2d_7x7):

    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vectors=2)
    include_2d = aplt.Include2D(
        origin=True,
        mask=True,
        border=True,
        light_profile_centres=True,
        mass_profile_centres=True,
        critical_curves=True,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_fit_imaging_from(fit=fit_imaging_x2_plane_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert (visuals_2d_via.mask == fit_imaging_x2_plane_7x7.mask).all()
    assert (
        visuals_2d_via.border == fit_imaging_x2_plane_7x7.mask.border_grid_sub_1.binned
    ).all()
    assert visuals_2d_via.light_profile_centres.in_list == [(0.0, 0.0)]
    assert visuals_2d_via.mass_profile_centres.in_list == [(0.0, 0.0)]
    assert (
        visuals_2d_via.critical_curves[0]
        == fit_imaging_x2_plane_7x7.tracer.critical_curves_from(grid=grid_2d_7x7)[0]
    ).all()
    assert visuals_2d_via.vectors == 2

    include_2d = aplt.Include2D(
        origin=False,
        mask=False,
        border=False,
        light_profile_centres=False,
        mass_profile_centres=False,
        critical_curves=False,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_fit_imaging_from(fit=fit_imaging_x2_plane_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert visuals_2d_via.mask is None
    assert visuals_2d_via.border is None
    assert visuals_2d_via.light_profile_centres is None
    assert visuals_2d_via.mass_profile_centres is None
    assert visuals_2d_via.critical_curves is None
    assert visuals_2d_via.vectors == 2
