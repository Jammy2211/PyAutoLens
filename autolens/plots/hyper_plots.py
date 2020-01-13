import autoarray as aa
from autoarray.plotters import plotters
from autoastro.plots import lensing_plotters
from autolens.plots.fit_imaging_plots import fit_imaging_plots


@plotters.set_subplot_filename
def subplot_fit_hyper_galaxy(
    fit,
    hyper_fit,
    galaxy_image,
    contribution_map_in,
    include=lensing_plotters.Include(),
    sub_plotter=plotters.SubPlotter(),
):

    number_subplots = 6

    sub_plotter.setup_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    hyper_galaxy_image(galaxy_image=galaxy_image, mask=include.mask_from_fit(fit=fit), plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 2)

    aa.plot.fit_imaging.noise_map(fit=fit, points=include.positions_from_fit(fit=fit), include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 3)

    aa.plot.fit_imaging.noise_map(fit=hyper_fit, points=include.positions_from_fit(fit=fit), include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 4)

    contribution_map(contribution_map_in=contribution_map_in, include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 5)

    aa.plot.fit_imaging.chi_squared_map(fit=fit, points=include.positions_from_fit(fit=fit), include=include, plotter=sub_plotter)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index= 6)

    aa.plot.fit_imaging.chi_squared_map(fit=hyper_fit, points=include.positions_from_fit(fit=fit), include=include, plotter=sub_plotter)

    sub_plotter.output.subplot_to_figure()

    sub_plotter.close_figure()

@plotters.set_subplot_filename
def subplot_hyper_galaxy_images(
    hyper_galaxy_image_path_dict, mask=None, include=lensing_plotters.Include(), sub_plotter=plotters.SubPlotter()
):

    number_subplots = 0

    for i in hyper_galaxy_image_path_dict.items():
        number_subplots += 1

    sub_plotter.setup_subplot_figure(number_subplots=number_subplots)

    hyper_index = 0

    for path, galaxy_image in hyper_galaxy_image_path_dict.items():

        hyper_index += 1

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=hyper_index)

        hyper_galaxy_image(
            galaxy_image=galaxy_image,
            mask=mask,
            plotter=sub_plotter,
        )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.close_figure()

@plotters.set_labels
def hyper_model_image(
    hyper_model_image,
    mask=None,
    positions=None,
    image_plane_pix_grid=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the image of a hyper_galaxies galaxy image.

    Set *autolens.datas.arrays.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_galaxy_image : datas.imaging.datas.Imaging
        The hyper_galaxies galaxy image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    plotter.array.plot(
        array=hyper_model_image, mask=mask, grid=image_plane_pix_grid, points=positions,
    )

@plotters.set_labels
def hyper_galaxy_image(
    galaxy_image,
    mask=None,
    positions=None,
    image_plane_pix_grid=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the image of a hyper_galaxies galaxy image.

    Set *autolens.datas.arrays.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_galaxy_image : datas.imaging.datas.Imaging
        The hyper_galaxies galaxy image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    plotter.array.plot(
        array=galaxy_image, mask=mask, grid=image_plane_pix_grid, points=positions,
    )

@plotters.set_labels
def contribution_map(
    contribution_map_in,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the summed contribution maps of a hyper_galaxies-fit.

    Set *autolens.datas.arrays.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The hyper_galaxies-fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the contribution_maps are plotted.
    """

    plotter.array.plot(
        array=contribution_map_in, mask=mask, points=positions,
    )
