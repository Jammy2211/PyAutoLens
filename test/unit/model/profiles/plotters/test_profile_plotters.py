import autolens as al
import pytest
import os


@pytest.fixture(name="profile_plotter_path")
def make_profile_plotter_setup():
    return "{}/../../../test_files/plotting/profiles/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__all_quantities_are_output(
    lp_0, mp_0, sub_grid_7x7, mask_7x7, positions_7x7, profile_plotter_path, plot_patch
):

    al.profile_plotters.plot_image(
        light_profile=lp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=profile_plotter_path,
        output_format="png",
    )

    assert profile_plotter_path + "image.png" in plot_patch.paths

    al.profile_plotters.plot_convergence(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        plot_critical_curves=False,
        plot_caustics=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=profile_plotter_path,
        output_format="png",
    )

    assert profile_plotter_path + "convergence.png" in plot_patch.paths

    al.profile_plotters.plot_potential(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        plot_critical_curves=False,
        plot_caustics=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=profile_plotter_path,
        output_format="png",
    )

    assert profile_plotter_path + "potential.png" in plot_patch.paths

    al.profile_plotters.plot_deflections_y(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        plot_critical_curves=False,
        plot_caustics=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=profile_plotter_path,
        output_format="png",
    )

    assert profile_plotter_path + "deflections_y.png" in plot_patch.paths

    al.profile_plotters.plot_deflections_x(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        plot_critical_curves=False,
        plot_caustics=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=profile_plotter_path,
        output_format="png",
    )

    assert profile_plotter_path + "deflections_x.png" in plot_patch.paths

    al.profile_plotters.plot_magnification(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        positions=positions_7x7,
        plot_critical_curves=False,
        plot_caustics=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=profile_plotter_path,
        output_format="png",
    )

    assert profile_plotter_path + "magnification.png" in plot_patch.paths
