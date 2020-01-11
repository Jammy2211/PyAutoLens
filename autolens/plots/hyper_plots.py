import autofit as af
import matplotlib

backend = af.conf.get_matplotlib_backend()
matplotlib.use(backend)

from matplotlib import pyplot as plt

from autoarray.plotters import plotters, array_plotters
from autoastro.plots import lensing_plotters
from autolens.plots.fit_imaging_plots import fit_imaging_plots


def subplot_of_hyper_galaxy(
    fit,
    hyper_fit,
    galaxy_image,
    include=lensing_plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):

    array_plotter = array_plotter.plotter_as_sub_plotter()
    array_plotter = array_plotter.plotter_with_new_output_filename(
        output_filename="hyper_galaxy_fit"
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

    hyper_galaxy_image(galaxy_image=galaxy_image, mask=include.mask_from_fit(fit=fit), array_plotter=array_plotter)

    plt.subplot(rows, columns, 2)

    fit_imaging_plots.noise_map(fit=fit, points=include.positions_from_fit(fit=fit), include=include, array_plotter=array_plotter)

    plt.subplot(rows, columns, 3)

    fit_imaging_plots.noise_map(fit=hyper_fit, points=include.positions_from_fit(fit=fit), include=include, array_plotter=array_plotter)

    plt.subplot(rows, columns, 4)

    fit_imaging_plots.contribution_maps(fit=hyper_fit, include=include, array_plotter=array_plotter)

    plt.subplot(rows, columns, 5)

    fit_imaging_plots.chi_squared_map(fit=fit, points=include.positions_from_fit(fit=fit), include=include, array_plotter=array_plotter)

    plt.subplot(rows, columns, 6)

    fit_imaging_plots.chi_squared_map(fit=hyper_fit, points=include.positions_from_fit(fit=fit), include=include, array_plotter=array_plotter)

    array_plotter.output.to_figure(structure=None, is_sub_plotter=False)

    plt.close()


def subplot_of_hyper_galaxy_images(
    hyper_galaxy_image_path_dict, mask=None, array_plotter=array_plotters.ArrayPlotter()
):

    array_plotter = array_plotter.plotter_as_sub_plotter()
    array_plotter = array_plotter.plotter_with_new_output_filename(
        output_filename="hyper_galaxy_fit"
    )

    rows, columns, figsize_tool = array_plotter.get_subplot_rows_columns_figsize(
        number_subplots=len(hyper_galaxy_image_path_dict)
    )

    if array_plotter.figsize is None:
        figsize = figsize_tool
    else:
        figsize = array_plotter.figsize

    plt.figure(figsize=figsize)

    hyper_index = 0

    for path, hyper_galaxy_image in hyper_galaxy_image_path_dict.items():

        hyper_index += 1

        plt.subplot(rows, columns, hyper_index)

        hyper_galaxy_image(
            galaxy_image=hyper_galaxy_image,
            mask=mask,
            array_plotter=array_plotter,
        )

    array_plotter.output.to_figure(structure=None, is_sub_plotter=False)

    plt.close()

@plotters.set_labels
def hyper_galaxy_image(
    galaxy_image,
    mask=None,
    positions=None,
    image_plane_pix_grid=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the image of a hyper_galaxies galaxy image.

    Set *autolens.datas.arrays.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_galaxy_image : datas.imaging.datas.Imaging
        The hyper_galaxies galaxy image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotter.plot_array(
        array=galaxy_image, mask=mask, grid=image_plane_pix_grid, points=positions,
    )

