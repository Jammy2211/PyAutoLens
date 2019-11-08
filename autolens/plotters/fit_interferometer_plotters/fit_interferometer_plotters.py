import autofit as af
import matplotlib

backend = af.conf.instance.visualize.get("figures", "backend", str)
matplotlib.use(backend)
from matplotlib import pyplot as plt

import autoarray as aa
from autoarray.plotters import plotter_util
from autolens.plotters import plane_plotters, ray_tracing_plotters


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
        number_subplots=9
    )

    if figsize is None:
        figsize = figsize_tool

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec

    image_plane_pix_grid = get_image_plane_pix_grid(include_image_plane_pix, fit)

    mask = get_mask(fit=fit, include_mask=include_mask)

    plt.figure(figsize=figsize)

    critical_curves, caustics = get_critical_curves_and_caustics(
        fit=fit,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    plt.subplot(rows, columns, 1)

    if not fit.tracer.has_pixelization:

        ray_tracing_plotters.profile_image(
            tracer=fit.tracer,
            grid=fit.masked_interferometer.grid,
            mask=mask,
            include_critical_curves=include_critical_curves,
            positions=positions,
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
            position_pointsize=position_pointsize,
            mask_pointsize=mask_pointsize,
            output_path=output_path,
            output_filename="",
            output_format=output_format,
        )

        plt.subplot(rows, columns, 2)

        plane_plotters.plane_image(
            plane=fit.tracer.source_plane,
            grid=fit.masked_interferometer.grid,
            as_subplot=True,
            lines=caustics,
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

    elif fit.tracer.has_pixelization:

        aa.plot.inversion.reconstructed_image(
            inversion=fit.inversion,
            mask=mask,
            lines=critical_curves,
            positions=positions,
            grid=image_plane_pix_grid,
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
            output_format=output_format,
            output_filename=output_filename,
        )

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

        plt.subplot(rows, columns, 2, aspect=float(aspect_inv))

        aa.plot.inversion.reconstruction(
            inversion=fit.inversion,
            lines=caustics,
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

    plt.subplot(rows, columns, 4)

    aa.plot.fit_interferometer.visibilities(
        fit=fit,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        cmap=cmap,
        cb_ticksize=cb_ticksize,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
        pointsize=grid_pointsize,
        output_path=output_path,
        output_filename="",
        output_format=output_format,
    )

    plt.subplot(rows, columns, 5)

    aa.plot.fit_interferometer.signal_to_noise_map(
        fit=fit,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        cmap=cmap,
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

    aa.plot.fit_interferometer.model_visibilities(
        fit=fit,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        cmap=cmap,
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

    plt.subplot(rows, columns, 7)

    aa.plot.fit_interferometer.residual_map(
        fit=fit,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        cmap=cmap,
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

    plt.subplot(rows, columns, 8)

    aa.plot.fit_interferometer.normalized_residual_map(
        fit=fit,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        cmap=cmap,
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

    plt.subplot(rows, columns, 9)

    aa.plot.fit_interferometer.chi_squared_map(
        fit=fit,
        as_subplot=True,
        units=units,
        kpc_per_arcsec=kpc_per_arcsec,
        figsize=figsize,
        cmap=cmap,
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


def individuals(
    fit,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_visibilities=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    plot_inversion_residual_map=False,
    plot_inversion_normalized_residual_map=False,
    plot_inversion_chi_squared_map=False,
    plot_inversion_regularization_weight_map=False,
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

    kpc_per_arcsec = fit.tracer.image_plane.kpc_per_arcsec

    if plot_visibilities:

        aa.plot.fit_interferometer.visibilities(
            fit=fit,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_noise_map:

        aa.plot.fit_interferometer.noise_map(
            fit=fit,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_signal_to_noise_map:

        aa.plot.fit_interferometer.signal_to_noise_map(
            fit=fit,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_model_visibilities:

        aa.plot.fit_interferometer.model_visibilities(
            fit=fit,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_residual_map:

        aa.plot.fit_interferometer.residual_map(
            fit=fit,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_normalized_residual_map:

        aa.plot.fit_interferometer.normalized_residual_map(
            fit=fit,
            units=units,
            kpc_per_arcsec=kpc_per_arcsec,
            output_path=output_path,
            output_format=output_format,
        )

    if plot_chi_squared_map:

        aa.plot.fit_interferometer.chi_squared_map(
            fit=fit,
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
        return fit.masked_interferometer.mask
    else:
        return None


def get_image_plane_pix_grid(include_image_plane_pix, fit):

    if fit.inversion is not None:
        if include_image_plane_pix and fit.inversion.mapper.is_image_plane_pixelization:
            return fit.tracer.sparse_image_plane_grids_of_planes_from_grid(
                grid=fit.grid
            )[-1]
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
            grid=fit.masked_interferometer.grid,
            include_critical_curves=include_critical_curves,
            include_caustics=False,
        )

        caustics = plotter_util.get_critical_curve_and_caustic(
            obj=fit.tracer,
            grid=fit.masked_interferometer.grid,
            include_critical_curves=False,
            include_caustics=include_caustics,
        )

    else:

        critical_curves = None
        caustics = None

    return critical_curves, caustics
