from autolens.model.profiles.plotters import profile_plotters
import pytest
import os


@pytest.fixture(name="profile_plotter_path")
def make_profile_plotter_setup():
    return "{}/../../../test_files/plotting/profiles/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__all_quantities_are_output(
    lp_0,
    mp_0,
    grid_stack_7x7,
    mask_7x7,
    positions_7x7,
    profile_plotter_path,
    plot_patch,
):

    profile_plotters.plot_intensities(
        light_profile=lp_0,
        grid=grid_stack_7x7.sub,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=profile_plotter_path,
        output_format="png",
    )

    assert profile_plotter_path + "intensities.png" in plot_patch.paths

    profile_plotters.plot_convergence(
        mass_profile=mp_0,
        grid=grid_stack_7x7.sub,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=profile_plotter_path,
        output_format="png",
    )

    assert profile_plotter_path + "convergence.png" in plot_patch.paths

    profile_plotters.plot_potential(
        mass_profile=mp_0,
        grid=grid_stack_7x7.sub,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=profile_plotter_path,
        output_format="png",
    )

    assert profile_plotter_path + "potential.png" in plot_patch.paths

    profile_plotters.plot_deflections_y(
        mass_profile=mp_0,
        grid=grid_stack_7x7.sub,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=profile_plotter_path,
        output_format="png",
    )

    assert profile_plotter_path + "deflections_y.png" in plot_patch.paths

    profile_plotters.plot_deflections_x(
        mass_profile=mp_0,
        grid=grid_stack_7x7.sub,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=profile_plotter_path,
        output_format="png",
    )

    assert profile_plotter_path + "deflections_x.png" in plot_patch.paths
