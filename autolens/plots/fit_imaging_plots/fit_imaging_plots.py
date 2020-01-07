import autofit as af
import matplotlib

backend = af.conf.get_matplotlib_backend()
matplotlib.use(backend)

import autoarray as aa
from autoarray.plotters import plotters, array_plotters, mapper_plotters
from autoarray.plots.fit_imaging_plots import *
from autoarray.util import plotter_util
from autoastro.plots import lens_plotter_util
from autolens.plots import plane_plots

@plotters.set_includes
def subplot(
    fit,
    mask=True,
    include_critical_curves=False,
    include_caustics=False,
    positions=False,
    include_image_plane_pix=False,
    array_plotter=array_plotters.ArrayPlotter(),
):

    image_plane_pix_grid = lens_plotter_util.get_image_plane_pix_grid_from_fit(
        include_image_plane_pix=include_image_plane_pix, fit=fit
    )

    positions = lens_plotter_util.get_positions_from_fit(fit=fit, positions=positions)

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
        mask=mask,
        grid=image_plane_pix_grid,
        lines=critical_curves,
        points=positions,
        array_plotter=array_plotter
    )

@plotters.set_includes
def subplot_of_planes(
    fit,
    mask=True,
    include_critical_curves=False,
    include_caustics=False,
    positions=False,
    include_image_plane_pix=False,
    array_plotter=array_plotters.ArrayPlotter(),
    mapper_plotter=mapper_plotters.MapperPlotter(),
):

    for plane_index in range(fit.tracer.total_planes):

        if (
            fit.tracer.planes[plane_index].has_light_profile
            or fit.tracer.planes[plane_index].has_pixelization
        ):

            subplot_for_plane(
                fit=fit,
                plane_index=plane_index,
                mask=mask,
                include_image_plane_pix=include_image_plane_pix,
                include_critical_curves=include_critical_curves,
                include_caustics=include_caustics,
                positions=positions,
                array_plotter=array_plotter,
                mapper_plotter=mapper_plotter
            )

@plotters.set_includes
def subplot_for_plane(
    fit,
    plane_index,
    mask=True,
    plot_source_grid=False,
    include_critical_curves=False,
    include_caustics=False,
    positions=False,
    include_image_plane_pix=False,
    include_mass_profile_centres=True,
    array_plotter=array_plotters.ArrayPlotter(),
    mapper_plotter=mapper_plotters.MapperPlotter(),
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

    mask = plotter_util.get_mask_from_fit(fit=fit, include_mask=mask)

    if figsize is None:
        figsize = figsize_tool

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.planes[plane_index], plot_in_kpc=plot_in_kpc
    )

    plt.figure(figsize=array_plotter.figsize)

    image_plane_pix_grid = lens_plotter_util.get_image_plane_pix_grid_from_fit(
        include_image_plane_pix=include_image_plane_pix, fit=fit
    )

    positions = lens_plotter_util.get_positions_from_fit(fit=fit, positions=positions)

    plt.subplot(rows, columns, 1)

    aa.plot.fit_imaging.image(
        fit=fit,
        mask=mask,
        grid=image_plane_pix_grid,
        points=positions,
        array_plotter=array_plotter)

    plt.subplot(rows, columns, 2)

    subtracted_image_of_plane(
        fit=fit,
        plane_index=plane_index,
        mask=mask,
        include_image_plane_pix=include_image_plane_pix,
        positions=positions,
        array_plotter=array_plotter
    )

    plt.subplot(rows, columns, 3)

    model_image_of_plane(
        fit=fit,
        plane_index=plane_index,
        mask=mask,
        positions=positions,
        include_mass_profile_centres=include_mass_profile_centres,
        include_critical_curves=include_critical_curves,
        array_plotter=array_plotter
    )

    caustics = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=fit.tracer, include_critical_curves=False, include_caustics=include_caustics
    )

    if not fit.tracer.planes[plane_index].has_pixelization:

        plt.subplot(rows, columns, 4)

        traced_grids = fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)

        plane_plots.plane_image(
            plane=fit.tracer.planes[plane_index],
            grid=traced_grids[plane_index],
            lines=caustics,
            include_grid=plot_source_grid,
            array_plotter=array_plotter
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

        if mapper_plotter.aspect is "square":
            aspect_inv = ratio
        elif mapper_plotter.aspect is "auto":
            aspect_inv = 1.0 / ratio
        elif mapper_plotter.aspect is "equal":
            aspect_inv = 1.0

        plt.subplot(rows, columns, 4, aspect=float(aspect_inv))

        aa.plot.inversion.reconstruction(
            inversion=fit.inversion,
            lines=caustics,
            include_grid=False,
            include_centres=False,
            mapper_plotter=mapper_plotter
        )

    array_plotter.output_subplot_array(
    )

    plt.close()

@plotters.set_includes
def individuals(
    fit,
    mask=True,
    positions=False,
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
    array_plotter=array_plotters.ArrayPlotter(),
    mapper_plotter=mapper_plotters.MapperPlotter(),
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

    positions = lens_plotter_util.get_positions_from_fit(fit=fit, positions=positions)

    critical_curves = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=fit.tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=False,
    )

    aa.plot.fit_imaging.individuals(
        fit=fit,
        mask=mask,
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
        array_plotter=array_plotter,
    )

    traced_grids = fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)

    if plot_subtracted_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            subtracted_image_of_plane(
                fit=fit,
                plane_index=plane_index,
                mask=mask,
                include_critical_curves=include_critical_curves,
                array_plotter=array_plotter
            )

    if plot_model_images_of_planes:

        for plane_index in range(fit.tracer.total_planes):

            model_image_of_plane(
                fit=fit,
                plane_index=plane_index,
                mask=mask,
                include_critical_curves=include_critical_curves,
                array_plotter=array_plotter
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

                plane_plots.plane_image(
                    plane=fit.tracer.planes[plane_index],
                    grid=traced_grids[plane_index],
                    lines=caustics,
                    array_plotter=array_plotter
                )

            elif fit.tracer.planes[plane_index].has_pixelization:

                aa.plot.inversion.reconstruction(
                    inversion=fit.inversion,
                    lines=caustics,
                    mapper_plotter=mapper_plotter,
                )

@plotters.set_includes
@plotters.set_labels
def subtracted_image_of_plane(
    fit,
    plane_index,
    mask=True,
    include_critical_curves=False,
    positions=False,
    include_image_plane_pix=False,
    array_plotter=array_plotters.ArrayPlotter(),
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

    mask = plotter_util.get_mask_from_fit(fit=fit, include_mask=mask)

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

    positions = lens_plotter_util.get_positions_from_fit(fit=fit, positions=positions)

    critical_curves = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=fit.tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=False,
    )

    array_plotter.plot_array(
        array=subtracted_image,
        mask=mask,
        grid=image_plane_pix_grid,
        points=positions,
        lines=critical_curves,
    )

@plotters.set_includes
@plotters.set_labels
def model_image_of_plane(
    fit,
    plane_index,
    mask=True,
    include_critical_curves=False,
    positions=False,
    include_mass_profile_centres=True,
    array_plotter=array_plotters.ArrayPlotter(),
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

    mask = plotter_util.get_mask_from_fit(fit=fit, include_mask=mask)

    centres = lens_plotter_util.get_mass_profile_centres_from_fit(
        include_mass_profile_centres=include_mass_profile_centres, fit=fit
    )

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    positions = lens_plotter_util.get_positions_from_fit(fit=fit, positions=positions)

    critical_curves = lens_plotter_util.get_critical_curves_and_caustics_from_lensing_object(
        obj=fit.tracer,
        include_critical_curves=include_critical_curves,
        include_caustics=False,
    )

    array_plotter.plot_array(
        array=fit.model_images_of_planes[plane_index],
        mask=mask,
        lines=critical_curves,
        points=positions,
        centres=centres,
    )

@plotters.set_includes
@plotters.set_labels
def contribution_maps(
    fit,
    mask=True,
    positions=False,
    array_plotter=array_plotters.ArrayPlotter(),
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

    mask = plotter_util.get_mask_from_fit(fit=fit, include_mask=mask)

    if len(fit.contribution_maps) > 1:
        contribution_map = sum(fit.contribution_maps)
    else:
        contribution_map = fit.contribution_maps[0]

    positions = lens_plotter_util.get_positions_from_fit(fit=fit, positions=positions)

    unit_label, unit_conversion_factor = lens_plotter_util.get_unit_label_and_unit_conversion_factor(
        obj=fit.tracer.image_plane, plot_in_kpc=plot_in_kpc
    )

    array_plotter.plot_array(
        array=contribution_map,
        mask=mask,
        points=positions,
    )
