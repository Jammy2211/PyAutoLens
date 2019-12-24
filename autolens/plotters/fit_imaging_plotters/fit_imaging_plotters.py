import autofit as af
import matplotlib

backend = af.conf.get_matplotlib_backend()
matplotlib.use(backend)
from matplotlib import pyplot as plt

import autoarray as aa
from autoarray.plotters.fit_imaging_plotters import *
from autoarray.util import plotter_util
from autoastro.plotters import lens_plotter_util
from autolens.plotters import plane_plotters


def subplot(
    fit,
    include_mask=True,
    include_critical_curves=False,
    include_caustics=False,
    include_positions=False,
    include_image_plane_pix=False,
    plot_in_kpc=False,
    figsize=None,
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_filename="fit",
    output_format="show",
):

    image_plane_pix_grid = lens_plotter_util.get_image_plane_pix_grid_from_fit(
        include_image_plane_pix=include_image_plane_pix, fit=fit
    )

    positions = lens_plotter_util.get_positions_from_fit(
        fit=fit, include_positions=include_positions
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    critical_curves = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=fit.tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=False,
    )

    aa.plot.fit_imaging.subplot(
        fit=fit,
        include_mask=include_mask,
        grid=image_plane_pix_grid,
        lines=critical_curves,
        points=positions,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        grid_pointsize=grid_pointsize,
        position_pointsize=position_pointsize,
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def subplot_of_planes(
    fit,
    include_mask=True,
    include_critical_curves=False,
    include_caustics=False,
    include_positions=False,
    include_image_plane_pix=False,
    plot_in_kpc=False,
    figsize=None,
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_filename="lens_fit_plane",
    output_format="show",
):

    for plane_index in range(fit.tracer.total_planes):

        if (
            fit.tracer.planes[plane_index].has_light_profile
            or fit.tracer.planes[plane_index].has_pixelization
        ):

            subplot_for_plane(
                fit=fit,
                plane_index=plane_index,
                include_mask=include_mask,
                include_image_plane_pix=include_image_plane_pix,
                include_critical_curves=include_critical_curves,
                include_caustics=include_caustics,
                include_positions=include_positions,
                plot_in_kpc=plot_in_kpc,
                figsize=figsize,
                aspect=aspect,
                cmap=cmap,
                norm=norm,
                norm_min=norm_min,
                norm_max=norm_max,
                linthresh=linthresh,
                linscale=linscale,
                cb_ticksize=cb_ticksize,
                cb_fraction=cb_fraction,
                cb_pad=cb_pad,
                cb_tick_values=cb_tick_values,
                cb_tick_labels=cb_tick_labels,
                titlesize=titlesize,
                xlabelsize=xlabelsize,
                ylabelsize=ylabelsize,
                xyticksize=xyticksize,
                grid_pointsize=grid_pointsize,
                position_pointsize=position_pointsize,
                mask_pointsize=mask_pointsize,
                output_path=output_path,
                output_filename=output_filename,
                output_format=output_format,
            )


def subplot_for_plane(
    fit,
    plane_index,
    include_mask=True,
    plot_source_grid=False,
    include_critical_curves=False,
    include_caustics=False,
    include_positions=False,
    include_image_plane_pix=False,
    include_mass_profile_centres=True,
    plot_in_kpc=False,
    figsize=None,
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    titlesize=10,
    xlabelsize=10,
    ylabelsize=10,
    xyticksize=10,
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_filename="lens_fit_plane",
    output_format="show",
):
    """Plot the model datas_ of an analysis, using the *Fitter* class object.

    The visualization and output type can be fully customized.

    Parameters
    -----------
    fit : autolens.lens.fitting.Fitter
        Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
    output_path : str
        The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
    output_filename : str
        The name of the file that is output, if the output_type is a file format (e.g. png, fits)
    output_format : str
        How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
        in the python interpreter window.
    """

    output_filename += "_" + str(plane_index)

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=4
    )

    mask = plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask)

    if figsize is None:
        figsize = figsize_tool

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.planes[plane_index], plot_in_kpc=plot_in_kpc
    )

    plt.figure(figsize=figsize)

    image_plane_pix_grid = lens_plotter_util.get_image_plane_pix_grid_from_fit(
        include_image_plane_pix=include_image_plane_pix, fit=fit
    )

    positions = lens_plotter_util.get_positions_from_fit(
        fit=fit, include_positions=include_positions
    )

    plt.subplot(rows, columns, 1)

    aa.plot.fit_imaging.image(
        fit=fit,
        mask=mask,
        grid=image_plane_pix_grid,
        points=positions,
        as_subplot=True,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        grid_pointsize=grid_pointsize,
        position_pointsize=position_pointsize,
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    subtracted_image_of_plane(
        fit=fit,
        plane_index=plane_index,
        include_mask=include_mask,
        include_image_plane_pix=include_image_plane_pix,
        include_positions=include_positions,
        as_subplot=True,
        plot_in_kpc=plot_in_kpc,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        xyticksize=xyticksize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 3)

    model_image_of_plane(
        fit=fit,
        plane_index=plane_index,
        include_mask=include_mask,
        include_positions=include_positions,
        include_mass_profile_centres=include_mass_profile_centres,
        include_critical_curves=include_critical_curves,
        as_subplot=True,
        plot_in_kpc=plot_in_kpc,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    caustics = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=fit.tracer, include_critical_curves=False, include_caustics=include_caustics
    )

    if not fit.tracer.planes[plane_index].has_pixelization:

        plt.subplot(rows, columns, 4)

        traced_grids = fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)

        plane_plotters.plane_image(
            plane=fit.tracer.planes[plane_index],
            grid=traced_grids[plane_index],
            lines=caustics,
            include_grid=plot_source_grid,
            as_subplot=True,
            plot_in_kpc=plot_in_kpc,
            figsize=figsize,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            grid_pointsize=grid_pointsize,
            position_pointsize=position_pointsize,
            output_path=output_path,
            output_filename="",
            output_format=output_format,
        )

    elif fit.tracer.planes[plane_index].has_pixelization:

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

        if aspect is "square":
            aspect_inv = ratio
        elif aspect is "auto":
            aspect_inv = 1.0 / ratio
        elif aspect is "equal":
            aspect_inv = 1.0

        plt.subplot(rows, columns, 4, aspect=float(aspect_inv))

        aa.plot.inversion.reconstruction(
            inversion=fit.inversion,
            lines=caustics,
            include_grid=False,
            include_centres=False,
            as_subplot=True,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            figsize=figsize,
            aspect=None,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            output_path=output_path,
            output_filename=None,
            output_format=output_format,
        )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def individuals(
    fit,
    include_mask=True,
    include_positions=False,
    include_critical_curves=False,
    include_caustics=False,
    include_image_plane_pix=False,
    plot_in_kpc=False,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_image=False,
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
    plot_subtracted_images_of_planes=False,
    plot_model_images_of_planes=False,
    plot_plane_images_of_planes=False,
    output_path=None,
    output_format="show",
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

    image_plane_pix_grid = lens_plotter_util.get_image_plane_pix_grid_from_fit(
        include_image_plane_pix=include_image_plane_pix, fit=fit
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    positions = lens_plotter_util.get_positions_from_fit(
        fit=fit, include_positions=include_positions
    )

    critical_curves = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=fit.tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=False,
    )

    aa.plot.fit_imaging.individuals(
        fit=fit,
        include_mask=include_mask,
        lines=critical_curves,
        grid=image_plane_pix_grid,
        points=positions,
        plot_image=plot_image,
        plot_noise_map=plot_noise_map,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        plot_model_image=plot_model_image,
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
        unit_label=unit_label,
        output_path=output_path,
        output_format=output_format,
    )

    traced_grids = fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)

    if plot_subtracted_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            subtracted_image_of_plane(
                fit=fit,
                plane_index=plane_index,
                include_mask=include_mask,
                include_critical_curves=include_critical_curves,
                plot_in_kpc=plot_in_kpc,
                output_path=output_path,
                output_format=output_format,
            )

    if plot_model_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            model_image_of_plane(
                fit=fit,
                plane_index=plane_index,
                include_mask=include_mask,
                include_critical_curves=include_critical_curves,
                plot_in_kpc=plot_in_kpc,
                output_path=output_path,
                output_format=output_format,
            )

    if plot_plane_images_of_planes:

        caustics = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
            obj=fit.tracer,
            include_critical_curves=False,
            include_caustics=include_caustics,
        )

        for plane_index in range(fit.tracer.total_planes):

            output_filename = "fit_plane_image_of_plane_" + str(plane_index)

            if fit.tracer.planes[plane_index].has_light_profile:

                plane_plotters.plane_image(
                    plane=fit.tracer.planes[plane_index],
                    grid=traced_grids[plane_index],
                    lines=caustics,
                    plot_in_kpc=plot_in_kpc,
                    output_path=output_path,
                    output_filename=output_filename,
                    output_format=output_format,
                )

            elif fit.tracer.planes[plane_index].has_pixelization:

                aa.plot.inversion.reconstruction(
                    inversion=fit.inversion,
                    lines=caustics,
                    unit_label=unit_label,
                    unit_conversion_factor=unit_conversion_factor,
                    output_path=output_path,
                    output_filename=output_filename,
                    output_format=output_format,
                )


def subtracted_image_of_plane(
    fit,
    plane_index,
    include_mask=True,
    include_critical_curves=False,
    include_positions=False,
    include_image_plane_pix=False,
    as_subplot=False,
    plot_in_kpc=False,
    figsize=(7, 7),
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Fit Subtracted Image Of Plane",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_subtracted_image_of_plane",
):
    """Plot the model image of a specific plane of a lens fit.

    Set *autolens.datas.arrays.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    plane_indexes : int
        The plane from which the model image is generated.
    """

    mask = plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask)

    output_filename += "_" + str(plane_index)

    if fit.tracer.total_planes > 1:

        other_planes_model_images = [
            model_image
            for i, model_image in enumerate(fit.model_images_of_planes)
            if i != plane_index
        ]

        subtracted_image = fit.image - sum(other_planes_model_images)

    else:

        subtracted_image = fit.image

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    image_plane_pix_grid = lens_plotter_util.get_image_plane_pix_grid_from_fit(
        include_image_plane_pix=include_image_plane_pix, fit=fit
    )

    positions = lens_plotter_util.get_positions_from_fit(
        fit=fit, include_positions=include_positions
    )

    critical_curves = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=fit.tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=False,
    )

    aa.plot.array(
        array=subtracted_image,
        mask=mask,
        grid=image_plane_pix_grid,
        points=positions,
        lines=critical_curves,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        point_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def model_image_of_plane(
    fit,
    plane_index,
    include_mask=True,
    include_critical_curves=False,
    include_positions=False,
    include_mass_profile_centres=True,
    as_subplot=False,
    plot_in_kpc=False,
    figsize=(7, 7),
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Model Image Of Plane",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_model_image_of_plane",
):
    """Plot the model image of a specific plane of a lens fit.

    Set *autolens.datas.arrays.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    plane_indexes : [int]
        The plane from which the model image is generated.
    """

    output_filename += "_" + str(plane_index)

    mask = plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask)

    centres = lens_plotter_util.get_mass_profile_centres_from_fit(
        include_mass_profile_centres=include_mass_profile_centres, fit=fit
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    positions = lens_plotter_util.get_positions_from_fit(
        fit=fit, include_positions=include_positions
    )

    critical_curves = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=fit.tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=False,
    )

    aa.plot.array(
        array=fit.model_images_of_planes[plane_index],
        mask=mask,
        lines=critical_curves,
        points=positions,
        centres=centres,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        point_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def contribution_maps(
    fit,
    include_mask=True,
    include_positions=False,
    as_subplot=False,
    plot_in_kpc=False,
    figsize=(7, 7),
    aspect="square",
    cmap="jet",
    norm="linear",
    norm_min=None,
    norm_max=None,
    linthresh=0.05,
    linscale=0.01,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    title="Contribution Map",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_contribution_maps",
):
    """Plot the summed contribution maps of a hyper_galaxies-fit.

    Set *autolens.datas.arrays.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The hyper_galaxies-fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the contribution_maps are plotted.
    """

    mask = plotter_util.get_mask_from_fit(fit=fit, include_mask=include_mask)

    if len(fit.contribution_maps) > 1:
        contribution_map = sum(fit.contribution_maps)
    else:
        contribution_map = fit.contribution_maps[0]

    positions = lens_plotter_util.get_positions_from_fit(
        fit=fit, include_positions=include_positions
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    aa.plot.array(
        array=contribution_map,
        mask=mask,
        points=positions,
        as_subplot=as_subplot,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        figsize=figsize,
        aspect=aspect,
        cmap=cmap,
        norm=norm,
        norm_min=norm_min,
        norm_max=norm_max,
        linthresh=linthresh,
        linscale=linscale,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        mask_pointsize=mask_pointsize,
        point_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )
