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
    include_mask=True,
    include_critical_curves=False,
    include_caustics=False,
    positions=None,
    include_image_plane_pix=False,
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
    mask_pointsize=10,
    position_pointsize=10,
    grid_pointsize=1,
    output_path=None,
    output_filename="fit",
    output_format="show",
):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    mask = get_mask(fit=fit, include_mask=include_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec

    image_plane_pix_grid = get_image_plane_pix_grid(include_image_plane_pix, fit)

    critical_curves, caustics = get_critical_curves_and_caustics(
        fit=fit,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    aa.plot.fit_imaging.image(
        fit=fit,
        grid=image_plane_pix_grid,
        mask=mask,
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
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    aa.plot.fit_imaging.signal_to_noise_map(
        fit=fit,
        mask=mask,
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
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 3)

    aa.plot.fit_imaging.model_image(
        fit=fit,
        mask=mask,
        lines=critical_curves,
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

    aa.plot.fit_imaging.residual_map(
        fit=fit,
        mask=mask,
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

    aa.plot.fit_imaging.normalized_residual_map(
        fit=fit,
        mask=mask,
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

    aa.plot.fit_imaging.chi_squared_map(
        fit=fit,
        mask=mask,
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
    include_mask=True,
    include_critical_curves=False,
    include_caustics=False,
    positions=None,
    include_image_plane_pix=False,
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
    positions=None,
    include_image_plane_pix=False,
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

    mask = get_mask(fit=fit, include_mask=include_mask)

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec

    image_plane_pix_grid = get_image_plane_pix_grid(include_image_plane_pix, fit)

    plt.subplot(rows, columns, 1)

    aa.plot.fit_imaging.image(
        fit=fit,
        mask=mask,
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
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 2)

    subtracted_image_of_plane(
        fit=fit,
        plane_index=plane_index,
        mask=mask,
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
        mask=mask,
        plot_mass_profile_centres=plot_mass_profile_centres,
        include_critical_curves=include_critical_curves,
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
        mask_pointsize=mask_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    if fit.tracer.has_mass_profile:

        lines = plotter_util.get_critical_curve_and_caustic(
            obj=fit.tracer,
            grid=fit.masked_imaging.grid,
            include_critical_curves=False,
            include_caustics=include_caustics,
        )

    else:

        lines = None

    if not fit.tracer.planes[plane_index].has_pixelization:

        plt.subplot(rows, columns, 4)

        traced_grids = fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)

        plane_plotters.plane_image(
            plane=fit.tracer.planes[plane_index],
            grid=traced_grids[plane_index],
            lines=lines,
            include_grid=plot_source_grid,
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
            lines=lines,
            include_grid=False,
            include_centres=False,
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
    include_mask=True,
    include_critical_curves=False,
    include_caustics=False,
    positions=None,
    include_image_plane_pix=False,
    plot_image=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_image=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_inversion_residual_map=False,
    plot_inversion_normalized_residual_map=False,
    plot_inversion_chi_squared_map=False,
    plot_inversion_regularization_weight_map=False,
    plot_subtracted_images_of_planes=False,
    plot_model_images_of_planes=False,
    plot_plane_images_of_planes=False,
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

    mask = get_mask(fit=fit, include_mask=include_mask)
    image_plane_pix_grid = get_image_plane_pix_grid(include_image_plane_pix, fit)

    critical_curves, caustics = get_critical_curves_and_caustics(
        fit=fit,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    traced_grids = fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec

    if plot_image:

        aa.plot.fit_imaging.image(
            fit=fit,
            mask=mask,
            positions=positions,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_noise_map:

        aa.plot.fit_imaging.noise_map(
            fit=fit,
            mask=mask,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_signal_to_noise_map:

        aa.plot.fit_imaging.signal_to_noise_map(
            fit=fit,
            mask=mask,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_model_image:

        aa.plot.fit_imaging.model_image(
            fit=fit,
            mask=mask,
            lines=critical_curves,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_residual_map:

        aa.plot.fit_imaging.residual_map(
            fit=fit,
            mask=mask,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_normalized_residual_map:

        aa.plot.fit_imaging.normalized_residual_map(
            fit=fit,
            mask=mask,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_chi_squared_map:

        aa.plot.fit_imaging.chi_squared_map(
            fit=fit,
            mask=mask,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_inversion_residual_map:

        if fit.total_inversions == 1:

            aa.plot.inversion.residual_map(
                inversion=fit.inversion,
                include_grid=True,
                units=units,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )

    if plot_inversion_normalized_residual_map:

        if fit.total_inversions == 1:

            aa.plot.inversion.normalized_residual_map(
                inversion=fit.inversion,
                include_grid=True,
                units=units,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )

    if plot_inversion_chi_squared_map:

        if fit.total_inversions == 1:

            aa.plot.inversion.chi_squared_map(
                inversion=fit.inversion,
                include_grid=True,
                units=units,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )

    if plot_inversion_regularization_weight_map:

        if fit.total_inversions == 1:

            aa.plot.inversion.regularization_weights(
                inversion=fit.inversion,
                include_grid=True,
                units=units,
                figsize=(20, 20),
                output_path=output_path,
                output_format=output_format,
            )

    if plot_subtracted_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            subtracted_image_of_plane(
                fit=fit,
                plane_index=plane_index,
                mask=mask,
                include_critical_curves=include_critical_curves,
                units=units,
                kpc_per_arcsec=kpc_per_arcsec,
                output_path=output_path,
                output_format=output_format,
            )

    if plot_model_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            model_image_of_plane(
                fit=fit,
                plane_index=plane_index,
                mask=mask,
                include_critical_curves=include_critical_curves,
                units=units,
                kpc_per_arcsec=kpc_per_arcsec,
                output_path=output_path,
                output_format=output_format,
            )

    if plot_plane_images_of_planes:

        if fit.tracer.has_mass_profile:

            lines = plotter_util.get_critical_curve_and_caustic(
                obj=fit.tracer,
                grid=fit.grid,
                include_critical_curves=False,
                include_caustics=include_caustics,
            )

        else:

            lines = None

        for plane_index in range(fit.tracer.total_planes):

            output_filename = "fit_plane_image_of_plane_" + str(plane_index)

            if fit.tracer.planes[plane_index].has_light_profile:

                plane_plotters.plane_image(
                    plane=fit.tracer.planes[plane_index],
                    grid=traced_grids[plane_index],
                    lines=lines,
                    units=units,
                    output_path=output_path,
                    output_filename=output_filename,
                    output_format=output_format,
                )

            elif fit.tracer.planes[plane_index].has_pixelization:

                aa.plot.inversion.reconstruction(
                    inversion=fit.inversion,
                    lines=lines,
                    units=units,
                    kpc_per_arcsec=kpc_per_arcsec,
                    output_path=output_path,
                    output_filename=output_filename,
                    output_format=output_format,
                )


def subtracted_image_of_plane(
    fit,
    plane_index,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
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

    if fit.tracer.has_mass_profile:

        lines = plotter_util.get_critical_curve_and_caustic(
            obj=fit.tracer,
            grid=fit.masked_imaging.grid,
            include_critical_curves=include_critical_curves,
            include_caustics=include_caustics,
        )

    else:

        lines = None

    aa.plot.array(
        array=subtracted_image,
        mask=mask,
        grid=image_plane_pix_grid,
        positions=positions,
        lines=lines,
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
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def model_image_of_plane(
    fit,
    plane_index,
    mask=None,
    include_critical_curves=False,
    include_caustics=False,
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

    centres = get_mass_profile_centes(
        plot_mass_profile_centres=plot_mass_profile_centres, fit=fit
    )

    if fit.tracer.has_mass_profile:

        lines = plotter_util.get_critical_curve_and_caustic(
            obj=fit.tracer,
            grid=fit.masked_imaging.grid,
            include_critical_curves=include_critical_curves,
            include_caustics=include_caustics,
        )

    else:

        lines = None

    aa.plot.array(
        array=fit.model_images_of_planes[plane_index],
        mask=mask,
        lines=lines,
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
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def contribution_maps(
    fit,
    mask=None,
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

    if len(fit.contribution_maps) > 1:
        contribution_map = sum(fit.contribution_maps)
    else:
        contribution_map = fit.contribution_maps[0]

    aa.plot.array(
        array=contribution_map,
        mask=mask,
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
        mask_pointsize=mask_pointsize,
        position_pointsize=position_pointsize,
        output_path=output_path,
        output_format=output_format,
        output_filename=output_filename,
    )


def get_image_plane_pix_grid(include_image_plane_pix, fit):

    if fit.inversion is not None:
        if include_image_plane_pix and fit.inversion.mapper.is_image_plane_pixelization:
            return fit.tracer.sparse_image_plane_grids_of_planes_from_grid(
                grid=fit.grid
            )[-1]
    else:
        return None


def get_mask(fit, include_mask):
    """Get the masks of the fit if the masks should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
    include_mask : bool
        If *True*, the masks is plotted on the fit's datas.
    """
    if include_mask:
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


def get_critical_curves_and_caustics(fit, include_critical_curves, include_caustics):
    if fit.tracer.has_mass_profile:

        critical_curves = plotter_util.get_critical_curve_and_caustic(
            obj=fit.tracer,
            grid=fit.masked_imaging.grid,
            include_critical_curves=include_critical_curves,
            include_caustics=False,
        )

        caustics = plotter_util.get_critical_curve_and_caustic(
            obj=fit.tracer,
            grid=fit.masked_imaging.grid,
            include_critical_curves=False,
            include_caustics=include_caustics,
        )

    else:

        critical_curves = None
        caustics = None

    return critical_curves, caustics
