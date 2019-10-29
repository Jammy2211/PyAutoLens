import autofit as af
import matplotlib

backend = af.conf.instance.visualize.get("figures", "backend", str)
matplotlib.use(backend)
from matplotlib import pyplot as plt

import autoarray as aa
from autoarray.plotters import plotter_util
from autolens.plotters import plane_plotters


def subplot(
    fit,
    should_plot_mask_overlay=True,
    positions=None,
    should_plot_image_plane_pix=False,
    plot_mass_profile_centres=True,
    units="arcsec",
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
    mask_overlay_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_filename="fit",
    output_format="show",
):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    mask_overlay = get_mask_overlay(
        fit=fit, should_plot_mask_overlay=should_plot_mask_overlay
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec

    image_plane_pix_grid = get_image_plane_pix_grid(should_plot_image_plane_pix, fit)

    aa.plot.imaging_fit.image(
        fit=fit,
        grid=image_plane_pix_grid,
        mask_overlay=mask_overlay,
        positions=positions,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        mask_overlay_pointsize=mask_overlay_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    aa.plot.imaging_fit.signal_to_noise_map(
        fit=fit,
        mask_overlay=mask_overlay,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        position_pointsize=position_pointsize,
        mask_overlay_pointsize=mask_overlay_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 3)

    aa.plot.imaging_fit.model_image(
        fit=fit,
        mask_overlay=mask_overlay,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 4)

    aa.plot.imaging_fit.residual_map(
        fit=fit,
        mask_overlay=mask_overlay,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 5)

    aa.plot.imaging_fit.normalized_residual_map(
        fit=fit,
        mask_overlay=mask_overlay,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 6)

    aa.plot.imaging_fit.chi_squared_map(
        fit=fit,
        mask_overlay=mask_overlay,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

    plt.close()


def subplot_of_planes(
    fit,
    should_plot_mask_overlay=True,
    positions=None,
    should_plot_image_plane_pix=False,
    plot_mass_profile_centres=True,
    units="arcsec",
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
    mask_overlay_pointsize=10,
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
                should_plot_mask_overlay=should_plot_mask_overlay,
                should_plot_image_plane_pix=should_plot_image_plane_pix,
                positions=positions,
                units=units,
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
                mask_overlay_pointsize=mask_overlay_pointsize,
                output_path=output_path,
                output_filename=output_filename,
                output_format=output_format,
            )


def subplot_for_plane(
    fit,
    plane_index,
    should_plot_mask_overlay=True,
    should_plot_source_grid=False,
    positions=None,
    should_plot_image_plane_pix=False,
    plot_mass_profile_centres=True,
    units="arcsec",
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
    mask_overlay_pointsize=10,
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

    mask_overlay = get_mask_overlay(
        fit=fit, should_plot_mask_overlay=should_plot_mask_overlay
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec

    image_plane_pix_grid = get_image_plane_pix_grid(should_plot_image_plane_pix, fit)

    plt.subplot(rows, columns, 1)

    aa.plot.imaging_fit.image(
        fit=fit,
        mask_overlay=mask_overlay,
        grid=image_plane_pix_grid,
        positions=positions,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        mask_overlay_pointsize=mask_overlay_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    subtracted_image_of_plane(
        fit=fit,
        plane_index=plane_index,
        mask_overlay=mask_overlay,
        image_plane_pix_grid=image_plane_pix_grid,
        positions=positions,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        mask_overlay_pointsize=mask_overlay_pointsize,
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
        mask_overlay=mask_overlay,
        plot_mass_profile_centres=plot_mass_profile_centres,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        mask_overlay_pointsize=mask_overlay_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    if not fit.tracer.planes[plane_index].has_pixelization:

        plt.subplot(rows, columns, 4)

        traced_grids = fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)

        plane_plotters.plane_image(
            plane=fit.tracer.planes[plane_index],
            grid=traced_grids[plane_index],
            positions=None,
            plot_grid=should_plot_source_grid,
            as_subplot=True,
            units=units,
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
                fit.inversion.mapper.grid.arc_second_maxima[1]
                - fit.inversion.mapper.grid.arc_second_minima[1]
            )
            / (
                fit.inversion.mapper.grid.arc_second_maxima[0]
                - fit.inversion.mapper.grid.arc_second_minima[0]
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
            positions=None,
            should_plot_grid=False,
            should_plot_centres=False,
            as_subplot=True,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
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
    should_plot_mask_overlay=True,
    positions=None,
    should_plot_image_plane_pix=False,
    should_plot_image=False,
    should_plot_noise_map=False,
    should_plot_signal_to_noise_map=False,
    should_plot_model_image=False,
    should_plot_residual_map=False,
    should_plot_normalized_residual_map=False,
    should_plot_chi_squared_map=False,
    should_plot_inversion_residual_map=False,
    should_plot_inversion_normalized_residual_map=False,
    should_plot_inversion_chi_squared_map=False,
    should_plot_inversion_regularization_weight_map=False,
    should_plot_subtracted_images_of_planes=False,
    should_plot_model_images_of_planes=False,
    should_plot_plane_images_of_planes=False,
    units="arcsec",
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

    mask_overlay = get_mask_overlay(
        fit=fit, should_plot_mask_overlay=should_plot_mask_overlay
    )
    image_plane_pix_grid = get_image_plane_pix_grid(should_plot_image_plane_pix, fit)

    traced_grids = fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec

    if should_plot_image:

        aa.plot.imaging_fit.image(
            fit=fit,
            mask_overlay=mask_overlay,
            positions=positions,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_noise_map:

        aa.plot.imaging_fit.noise_map(
            fit=fit,
            mask_overlay=mask_overlay,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_signal_to_noise_map:

        aa.plot.imaging_fit.signal_to_noise_map(
            fit=fit,
            mask_overlay=mask_overlay,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_model_image:

        aa.plot.imaging_fit.model_image(
            fit=fit,
            mask_overlay=mask_overlay,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_residual_map:

        aa.plot.imaging_fit.residual_map(
            fit=fit,
            mask_overlay=mask_overlay,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_normalized_residual_map:

        aa.plot.imaging_fit.normalized_residual_map(
            fit=fit,
            mask_overlay=mask_overlay,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_chi_squared_map:

        aa.plot.imaging_fit.chi_squared_map(
            fit=fit,
            mask_overlay=mask_overlay,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if should_plot_inversion_residual_map:

        if fit.total_inversions == 1:

            aa.plot.inversion.residual_map(
                inversion=fit.inversion,
                should_plot_grid=True,
                units=units,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )

    if should_plot_inversion_normalized_residual_map:

        if fit.total_inversions == 1:

            aa.plot.inversion.normalized_residual_map(
                inversion=fit.inversion,
                should_plot_grid=True,
                units=units,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )

    if should_plot_inversion_chi_squared_map:

        if fit.total_inversions == 1:

            aa.plot.inversion.chi_squared_map(
                inversion=fit.inversion,
                should_plot_grid=True,
                units=units,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )

    if should_plot_inversion_regularization_weight_map:

        if fit.total_inversions == 1:

            aa.plot.inversion.regularization_weights(
                inversion=fit.inversion,
                should_plot_grid=True,
                units=units,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )

    if should_plot_subtracted_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            subtracted_image_of_plane(
                fit=fit,
                plane_index=plane_index,
                mask_overlay=mask_overlay,
                units=units,
                kpc_per_arcsec=kpc_per_arcsec,
                output_path=output_path,
                output_format=output_format,
            )

    if should_plot_model_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            model_image_of_plane(
                fit=fit,
                plane_index=plane_index,
                mask_overlay=mask_overlay,
                units=units,
                kpc_per_arcsec=kpc_per_arcsec,
                output_path=output_path,
                output_format=output_format,
            )

    if should_plot_plane_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            if fit.tracer.planes[plane_index].has_light_profile:

                output_filename = "fit_plane_image_of_plane_" + str(plane_index)

                plane_plotters.plane_image(
                    plane=fit.tracer.planes[plane_index],
                    grid=traced_grids[plane_index],
                    plot_grid=True,
                    units=units,
                    output_path=output_path,
                    output_filename=output_filename,
                    output_format=output_format,
                )


def subtracted_image_of_plane(
    fit,
    plane_index,
    mask_overlay=None,
    positions=None,
    image_plane_pix_grid=None,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
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
    title="Fit Model Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_overlay_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_subtracted_image_of_plane",
):
    """Plot the model image of a specific plane of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the model image is plotted.
    plane_indexes : int
        The plane from which the model image is generated.
    """

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

    aa.plot.array(
        array=subtracted_image,
        mask_overlay=mask_overlay,
        grid=image_plane_pix_grid,
        positions=positions,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        mask_overlay_pointsize=mask_overlay_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def model_image_of_plane(
    fit,
    plane_index,
    mask_overlay=None,
    positions=None,
    plot_mass_profile_centres=True,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
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
    title="Model Image",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_overlay_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_model_image_of_plane",
):
    """Plot the model image of a specific plane of a lens fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    plane_indexes : [int]
        The plane from which the model image is generated.
    """

    output_filename += "_" + str(plane_index)

    centres = get_mass_profile_centes(
        plot_mass_profile_centres=plot_mass_profile_centres, fit=fit
    )

    aa.plot.array(
        array=fit.model_images_of_planes[plane_index],
        mask_overlay=mask_overlay,
        positions=positions,
        centres=centres,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        mask_overlay_pointsize=mask_overlay_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def contribution_maps(
    fit,
    mask_overlay=None,
    positions=None,
    as_subplot=False,
    units="arcsec",
    kpc_per_arcsec=None,
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
    title="Contributions",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    xyticksize=16,
    mask_overlay_pointsize=10,
    position_pointsize=10,
    output_path=None,
    output_format="show",
    output_filename="fit_contribution_maps",
):
    """Plot the summed contribution maps of a hyper_galaxies-fit.

    Set *autolens.datas.array.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The hyper_galaxies-fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the contribution_maps are plotted.
    """

    if len(fit.contribution_maps) > 1:
        contribution_map = sum(fit.contribution_maps)
    else:
        contribution_map = fit.contribution_maps[0]

    aa.plot.array(
        array=contribution_map,
        mask_overlay=mask_overlay,
        positions=positions,
        as_subplot=as_subplot,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
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
        mask_overlay_pointsize=mask_overlay_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def get_image_plane_pix_grid(should_plot_image_plane_pix, fit):

    if fit.inversion is not None:
        if (
            should_plot_image_plane_pix
            and fit.inversion.mapper.is_image_plane_pixelization
        ):
            return fit.tracer.sparse_image_plane_grids_of_planes_from_grid(grid=fit.grid)[-1]
    else:
        return None


def get_mask_overlay(fit, should_plot_mask_overlay):
    """Get the mask_overlays of the fit if the mask_overlays should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
    should_plot_mask_overlay : bool
        If *True*, the mask_overlays is plotted on the fit's datas.
    """
    if should_plot_mask_overlay:
        return fit.mask
    else:
        return None


def get_mass_profile_centes(plot_mass_profile_centres, fit):

    if not hasattr(fit, "tracer"):
        return None

    if plot_mass_profile_centres:
        return fit.tracer.image_plane.centres_of_galaxy_mass_profiles
    else:
        return None
