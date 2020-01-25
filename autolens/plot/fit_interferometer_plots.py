from autoarray.plot import plotters, mat_objs
from autoastro.plot import lensing_plotters
from autolens.plot import plane_plots, ray_tracing_plots, inversion_plots


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_fit_interferometer(fit, include=None, sub_plotter=None):

    number_subplots = 6

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    residual_map_vs_uv_distances(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    normalized_residual_map_vs_uv_distances(
        fit=fit, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=3)

    chi_squared_map_vs_uv_distances(fit=fit, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=4)

    residual_map_vs_uv_distances(
        fit=fit, plot_real=False, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=5)

    normalized_residual_map_vs_uv_distances(
        fit=fit, plot_real=False, include=include, plotter=sub_plotter
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=6)

    chi_squared_map_vs_uv_distances(
        fit=fit, plot_real=False, include=include, plotter=sub_plotter
    )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_fit_real_space(fit, include=None, sub_plotter=None):

    number_subplots = 2

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    if fit.inversion is None:

        ray_tracing_plots.profile_image(
            tracer=fit.tracer,
            grid=fit.masked_interferometer.grid,
            positions=include.positions_from_fit(fit=fit),
            include=include,
            plotter=sub_plotter,
        )

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        plane_plots.plane_image(
            plane=fit.tracer.source_plane,
            grid=fit.masked_interferometer.grid,
            positions=include.positions_of_plane_from_fit_and_plane_index(
                fit=fit, plane_index=-1
            ),
            caustics=include.caustics_from_obj(obj=fit.tracer),
            plotter=sub_plotter,
        )

    elif fit.inversion is not None:

        inversion_plots.reconstructed_image(
            inversion=fit.inversion,
            light_profile_centres=include.light_profile_centres_of_galaxies_from_obj(
                fit.tracer.image_plane
            ),
            mass_profile_centres=include.mass_profile_centres_of_galaxies_from_obj(
                fit.tracer.image_plane
            ),
            critical_curves=include.critical_curves_from_obj(obj=fit.tracer),
            image_positions=include.positions_from_fit(fit=fit),
            grid=include.inversion_image_pixelization_grid_from_fit(fit=fit),
            plotter=sub_plotter,
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

        if sub_plotter.figure.aspect in "square":
            aspect_inv = ratio
        elif sub_plotter.figure.aspect in "auto":
            aspect_inv = 1.0 / ratio
        elif sub_plotter.figure.aspect in "equal":
            aspect_inv = 1.0

        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=2, aspect=float(aspect_inv)
        )

        inversion_plots.reconstruction(
            inversion=fit.inversion,
            source_positions=include.positions_of_plane_from_fit_and_plane_index(
                fit=fit, plane_index=-1
            ),
            caustics=include.caustics_from_obj(obj=fit.tracer),
            include=include,
            plotter=sub_plotter,
        )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


def individuals(
    fit,
    plot_visibilities=False,
    plot_noise_map=False,
    plot_signal_to_noise_map=False,
    plot_model_visibilities=False,
    plot_residual_map=False,
    plot_normalized_residual_map=False,
    plot_chi_squared_map=False,
    include=None,
    plotter=None,
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

    if plot_visibilities:

        visibilities(fit=fit, include=include, plotter=plotter)

    if plot_noise_map:

        noise_map(fit=fit, include=include, plotter=plotter)

    if plot_signal_to_noise_map:

        signal_to_noise_map(fit=fit, include=include, plotter=plotter)

    if plot_model_visibilities:

        model_visibilities(fit=fit, include=include, plotter=plotter)

    if plot_residual_map:

        residual_map_vs_uv_distances(
            fit=fit, plot_real=True, include=include, plotter=plotter
        )

        residual_map_vs_uv_distances(
            fit=fit, plot_real=False, include=include, plotter=plotter
        )

    if plot_normalized_residual_map:

        normalized_residual_map_vs_uv_distances(
            fit=fit, plot_real=True, include=include, plotter=plotter
        )

        normalized_residual_map_vs_uv_distances(
            fit=fit, plot_real=False, include=include, plotter=plotter
        )

    if plot_chi_squared_map:

        chi_squared_map_vs_uv_distances(
            fit=fit, plot_real=True, include=include, plotter=plotter
        )

        chi_squared_map_vs_uv_distances(
            fit=fit, plot_real=False, include=include, plotter=plotter
        )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def visibilities(fit, include=None, plotter=None):
    """Plot the visibilities of a lens fit.

    Set *autolens.datas.grid.lensing_plotters.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which include the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    plotter.plot_grid(grid=fit.visibilities)


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def noise_map(fit, include=None, plotter=None):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.lensing_plotters.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
        The datas-datas, which include the observed datas, noise_map-map, PSF, signal-to-noise_map-map, etc.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    plotter.plot_grid(grid=fit.visibilities, color_array=fit.noise_map[:, 0])


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def signal_to_noise_map(fit, include=None, plotter=None):
    """Plot the noise-map of a lens fit.

    Set *autolens.datas.grid.lensing_plotters.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    visibilities : datas.imaging.datas.Imaging
    The datas-datas, which include the observed datas, signal_to_noise_map-map, PSF, signal-to-signal_to_noise_map-map, etc.
    origin : True
    If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """
    plotter.plot_grid(grid=fit.visibilities, color_array=fit.signal_to_noise_map[:, 0])


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def model_visibilities(fit, include=None, plotter=None):
    """Plot the model visibilities of a fit.

    Set *autolens.datas.grid.lensing_plotters.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the model visibilities is plotted.
    """
    plotter.plot_grid(grid=fit.visibilities)


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def residual_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    include=None,
    plotter=None,
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.lensing_plotters.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = fit.residual_map[:, 0]
        plotter = plotter.plotter_with_new_labels(title=plotter.labels.title + " Real")
        plotter = plotter.plotter_with_new_output(
            filename=plotter.output.filename + "_real"
        )
    else:
        y = fit.residual_map[:, 1]
        plotter = plotter.plotter_with_new_labels(title=plotter.labels.title + " Imag")
        plotter = plotter.plotter_with_new_output(
            filename=plotter.output.filename + "_imag"
        )

    plotter.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def normalized_residual_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    include=None,
    plotter=None,
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.lensing_plotters.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = fit.residual_map[:, 0]
        plotter = plotter.plotter_with_new_labels(title=plotter.labels.title + " Real")
        plotter = plotter.plotter_with_new_output(
            filename=plotter.output.filename + "_real"
        )
    else:
        y = fit.residual_map[:, 1]
        plotter = plotter.plotter_with_new_labels(title=plotter.labels.title + " Imag")
        plotter = plotter.plotter_with_new_output(
            filename=plotter.output.filename + "_imag"
        )

    plotter.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def chi_squared_map_vs_uv_distances(
    fit,
    plot_real=True,
    label_yunits="V$_{R,data}$ - V$_{R,model}$",
    label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    include=None,
    plotter=None,
):
    """Plot the residual-map of a lens fit.

    Set *autolens.datas.grid.lensing_plotters.Plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractFitter
        The fit to the datas, which include a list of every model visibilities, residual_map, chi-squareds, etc.
    visibilities_index : int
        The index of the datas in the datas-set of which the residual_map are plotted.
    """

    if plot_real:
        y = fit.residual_map[:, 0]
        plotter = plotter.plotter_with_new_labels(title=plotter.labels.title + " Real")
        plotter = plotter.plotter_with_new_output(
            filename=plotter.output.filename + "_real"
        )
    else:
        y = fit.residual_map[:, 1]
        plotter = plotter.plotter_with_new_labels(title=plotter.labels.title + " Imag")
        plotter = plotter.plotter_with_new_output(
            filename=plotter.output.filename + "_imag"
        )

    plotter.plot_line(
        y=y,
        x=fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
        plot_axis_type="scatter",
    )
