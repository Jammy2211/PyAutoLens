import autofit as af
import matplotlib

backend = af.conf.get_matplotlib_backend()
matplotlib.use(backend)

import autoarray as aa
from autoarray.plots.fit_interferometer_plots import *
from autoarray.plotters import plotters, array_plotters, grid_plotters, line_plotters, mapper_plotters
from autoarray.util import plotter_util
from autoastro.plots import lens_plotter_util
from autolens.plots import plane_plots, ray_tracing_plots


def subplot(
    fit,
    array_plotter=array_plotters.ArrayPlotter(),
    grid_plotter=grid_plotters.GridPlotter(),
    line_plotter=line_plotters.LinePlotter(),
):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    if figsize is None:
        figsize = figsize_tool

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    plt.figure(figsize=figsize)

    plt.subplot(rows, columns, 1)

    aa.plot.fit_interferometer.subplot(
        fit=fit,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        array_plotter=array_plotter,
        grid_plotter=grid_plotter,
        line_plotter=line_plotter,
    )


def subplot_real_space(
    fit,
    mask=True,
    include_critical_curves=False,
    include_caustics=False,
    positions=True,
    include_image_plane_pix=False,
    include_mass_profile_centres=True,
    array_plotter=array_plotters.ArrayPlotter(),
    mapper_plotter=mapper_plotters.MapperPlotter(),
):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=2
    )

    if figsize is None:
        figsize = figsize_tool

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    image_plane_pix_grid = lens_plotter_util.get_image_plane_pix_grid_from_fit(
        include_image_plane_pix=include_image_plane_pix, fit=fit
    )

    real_space_mask = plotter_util.get_real_space_mask_from_fit(fit=fit, mask=mask)

    positions = lens_plotter_util.get_positions_from_fit(fit=fit, positions=positions)

    plt.figure(figsize=figsize)

    lines = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=fit.tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    plt.subplot(rows, columns, 1)

    if not fit.inversion is not None:

        ray_tracing_plots.profile_image(
            tracer=fit.tracer,
            grid=fit.masked_interferometer.grid,
            mask=real_space_mask,
            include_critical_curves=include_critical_curves,
            positions=positions,
        )

        plt.subplot(rows, columns, 2)

        plane_plots.plane_image(
            plane=fit.tracer.source_plane,
            grid=fit.masked_interferometer.grid,
            as_subplot=True,
            lines=[lines[1]],
        )

    elif fit.inversion is not None:

        aa.plot.inversion.reconstructed_image(
            inversion=fit.inversion,
            mask=real_space_mask,
            lines=[lines[0]],
            positions=positions,
            grid=image_plane_pix_grid,
        )

        ratio = float(
            (
                fit.inversion.mapper.grid.scaled_maxima[1]
                - fit.inversion.mapper.grid.scaled_minima[1]
            )
            / (
                fit.inversion.mapper.grid.scaled_maxima[0]
                - fit.inversion.mapper.grid.scaled_minima[0]
            )
        )

        if mapper_plotter.aspect is "square":
            aspect_inv = ratio
        elif mapper_plotter.aspect is "auto":
            aspect_inv = 1.0 / ratio
        elif mapper_plotter.aspect is "equal":
            aspect_inv = 1.0

        plt.subplot(rows, columns, 2, aspect=float(aspect_inv))

        aa.plot.inversion.reconstruction(
            inversion=fit.inversion,
            lines=[lines[0]],
            include_grid=False,
            include_centres=False,
            mapper_plotter=mapper_plotter
        )

    array_plotter.output_subplot_array(
    )

    plt.close()


def individuals(
    fit,
    plot_in_kpc=False,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_visibilities=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_inversion_reconstruction=False,
    plot_inversion_errors=False,
    plot_inversion_residual_map=False,
    plot_inversion_normalized_residual_map=False,
    plot_inversion_chi_squared_map=False,
    plot_inversion_regularization_weight_map=False,
    plot_inversion_interpolated_reconstruction=False,
    plot_inversion_interpolated_errors=False,
    array_plotter=array_plotters.ArrayPlotter(),
    grid_plotter=grid_plotters.GridPlotter(),
    line_plotter=line_plotters.LinePlotter(),
):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    aa.plot.fit_interferometer.individuals(
        fit=fit,
        plot_visibilities=plot_visibilities,
        plot_noise_map=plot_noise_map,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        plot_model_visibilities=plot_model_visibilities,
        plot_residual_map=plot_residual_map,
        plot_normalized_residual_map=plot_normalized_residual_map,
        plot_chi_squared_map=plot_chi_squared_map,
        plot_inversion_reconstruction=plot_inversion_reconstruction,
        plot_inversion_errors=plot_inversion_errors,
        plot_inversion_residual_map=plot_inversion_residual_map,
        plot_inversion_normalized_residual_map=plot_inversion_normalized_residual_map,
        plot_inversion_chi_squared_map=plot_inversion_chi_squared_map,
        plot_inversion_regularization_weight_map=plot_inversion_regularization_weight_map,
        plot_inversion_interpolated_reconstruction=plot_inversion_interpolated_reconstruction,
        plot_inversion_interpolated_errors=plot_inversion_interpolated_errors,
        unit_conversion_factor=unit_conversion_factor,
        array_plotter=array_plotter,
        grid_plotter=grid_plotter,
        line_plotter=line_plotter,
    )
