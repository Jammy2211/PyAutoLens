import autofit as af
import matplotlib

backend = af.conf.get_matplotlib_backend()
matplotlib.use(backend)

import autoarray as aa
from autoarray.plots.fit_interferometer_plots import *
from autoarray.plotters import (
    plotters,
    array_plotters,
    grid_plotters,
    line_plotters,
    mapper_plotters,
)
from autoastro.plots import lensing_plotters
from autolens.plots import plane_plots, ray_tracing_plots


def subplot(
    fit,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
    grid_plotter=grid_plotters.GridPlotter(),
    line_plotter=line_plotters.LinePlotter(),
):

    array_plotter = array_plotter.plotter_as_sub_plotter()
    array_plotter = array_plotter.plotter_with_new_output_filename(
        output_filename="fit_interferometer"
    )

    rows, columns, figsize_tool = array_plotter.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    if array_plotter.figsize is None:
        figsize = figsize_tool
    else:
        figsize = array_plotter.figsize

    plt.figure(figsize=figsize)

    plt.subplot(rows, columns, 1)

    aa.plot.fit_interferometer.subplot(
        fit=fit,
        include=include,
        array_plotter=array_plotter,
        grid_plotter=grid_plotter,
        line_plotter=line_plotter,
    )


def subplot_real_space(
    fit,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
    mapper_plotter=mapper_plotters.MapperPlotter(),
):

    array_plotter = array_plotter.plotter_as_sub_plotter()
    array_plotter = array_plotter.plotter_with_new_output_filename(
        output_filename="fit_real_space"
    )

    rows, columns, figsize_tool = array_plotter.get_subplot_rows_columns_figsize(
        number_subplots=2
    )

    if array_plotter.figsize is None:
        figsize = figsize_tool
    else:
        figsize = array_plotter.figsize

    plt.figure(figsize=figsize)

    plt.subplot(rows, columns, 1)

    if fit.inversion is None:

        ray_tracing_plots.profile_image(
            tracer=fit.tracer,
            grid=fit.masked_interferometer.grid,
            mask=include.real_space_mask_from_fit(fit=fit),
            positions=include.positions_from_fit(fit=fit),
            include=include,
            array_plotter=array_plotter
        )

        plt.subplot(rows, columns, 2)

        plane_plots.plane_image(
            plane=fit.tracer.source_plane,
            grid=fit.masked_interferometer.grid,
            lines=include.caustics_from_obj(obj=fit.tracer),
            array_plotter=array_plotter
        )

    elif fit.inversion is not None:

        aa.plot.inversion.reconstructed_image(
            inversion=fit.inversion,
            mask=include.real_space_mask_from_fit(fit=fit),
            lines=include.critical_curves_from_obj(obj=fit.tracer),
            positions=include.positions_from_fit(fit=fit),
            grid=include.inversion_image_pixelization_grid_from_fit(fit=fit),
            array_plotter=array_plotter,
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
            lines=include.caustics_from_obj(obj=fit.tracer),
            include=include,
            mapper_plotter=mapper_plotter,
        )

    array_plotter.output.to_figure(structure=None, bypass=False)

    plt.close()

def individuals(
    fit,
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
    include=lensing_plotters.Include(),
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
        include=include,
        array_plotter=array_plotter,
        grid_plotter=grid_plotter,
        line_plotter=line_plotter,
    )
