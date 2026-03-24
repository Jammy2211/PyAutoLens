"""Standalone subplot functions for subhalo detection visualisation."""
import matplotlib.pyplot as plt
from typing import Optional

from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.utils import save_figure
from autolens.imaging.plot.fit_imaging_plots import _plot_source_plane


def subplot_detection_imaging(
    result,
    fit_imaging_with_subhalo,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    use_log10: bool = False,
    use_log_evidences: bool = True,
    relative_to_value: float = 0.0,
    remove_zeros: bool = False,
):
    """4-panel subplot: data, S/N map, log-evidence increase, subhalo mass grid."""
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    plot_array(
        array=fit_imaging_with_subhalo.data,
        ax=axes[0],
        title="Data",
        colormap=colormap,
        use_log10=use_log10,
    )
    plot_array(
        array=fit_imaging_with_subhalo.signal_to_noise_map,
        ax=axes[1],
        title="Signal-To-Noise Map",
        colormap=colormap,
        use_log10=use_log10,
    )

    fom_array = result.figure_of_merit_array(
        use_log_evidences=use_log_evidences,
        relative_to_value=relative_to_value,
        remove_zeros=remove_zeros,
    )
    plot_array(
        array=fom_array,
        ax=axes[2],
        title="Increase in Log Evidence",
        colormap=colormap,
    )

    mass_array = result.subhalo_mass_array
    plot_array(
        array=mass_array,
        ax=axes[3],
        title="Subhalo Mass",
        colormap=colormap,
    )

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_detection_imaging", format=output_format)


def subplot_detection_fits(
    fit_imaging_no_subhalo,
    fit_imaging_with_subhalo,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
):
    """6-panel subplot comparing fits with and without a subhalo."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    plot_array(
        array=fit_imaging_no_subhalo.normalized_residual_map,
        ax=axes[0][0],
        title="Normalized Residual Map (No Subhalo)",
        colormap=colormap,
    )
    plot_array(
        array=fit_imaging_no_subhalo.chi_squared_map,
        ax=axes[0][1],
        title="Chi-Squared Map (No Subhalo)",
        colormap=colormap,
    )
    _plot_source_plane(fit_imaging_no_subhalo, axes[0][2], plane_index=1,
                       colormap=colormap)

    plot_array(
        array=fit_imaging_with_subhalo.normalized_residual_map,
        ax=axes[1][0],
        title="Normalized Residual Map (With Subhalo)",
        colormap=colormap,
    )
    plot_array(
        array=fit_imaging_with_subhalo.chi_squared_map,
        ax=axes[1][1],
        title="Chi-Squared Map (With Subhalo)",
        colormap=colormap,
    )
    _plot_source_plane(fit_imaging_with_subhalo, axes[1][2], plane_index=1,
                       colormap=colormap)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_detection_fits", format=output_format)
