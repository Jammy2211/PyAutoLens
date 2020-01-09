import autofit as af
import matplotlib

backend = af.conf.get_matplotlib_backend()
matplotlib.use(backend)

from matplotlib import pyplot as plt

import autoarray as aa
from autoarray.plotters import plotters, array_plotters
from autoarray.util import plotter_util


def subplot_of_hyper_galaxy(
    hyper_galaxy_image_sub,
    contribution_map_sub,
    noise_map_sub,
    hyper_noise_map_sub,
    chi_squared_map_sub,
    hyper_chi_squared_map_sub,
    mask=None,
    array_plotter=array_plotters.ArrayPlotter(),
):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=6
    )

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    plt.subplot(rows, columns, 1)

    hyper_galaxy_image(hyper_galaxy_image=hyper_galaxy_image_sub, mask=mask)

    plt.subplot(rows, columns, 2)

    array_plotter.plot_array(array=noise_map_sub, mask=mask)

    plt.subplot(rows, columns, 3)

    hyper_noise_map(hyper_noise_map=hyper_noise_map_sub, mask=mask)

    plt.subplot(rows, columns, 4)

    contribution_map(contribution_map=contribution_map_sub, mask=mask)

    plt.subplot(rows, columns, 5)

    chi_squared_map(chi_squared_map=chi_squared_map_sub, mask=mask)

    plt.subplot(rows, columns, 6)

    hyper_chi_squared_map(hyper_chi_squared_map=hyper_chi_squared_map_sub, mask=mask)

    array_plotter.output.to_figure(structure=None, is_sub_plotter=False)

    plt.close()


def subplot_of_hyper_galaxy_images(
    hyper_galaxy_image_path_dict, mask=True, array_plotter=array_plotters.ArrayPlotter()
):

    rows, columns, figsize_tool = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=len(hyper_galaxy_image_path_dict)
    )

    if not mask:
        mask = False

    if figsize is None:
        figsize = figsize_tool

    plt.figure(figsize=figsize)

    hyper_index = 0

    for path, hyper_galaxy_image_sub in hyper_galaxy_image_path_dict.items():

        hyper_index += 1

        plt.subplot(rows, columns, hyper_index)

        hyper_galaxy_image(
            hyper_galaxy_image=hyper_galaxy_image_sub,
            mask=mask,
            array_plotter=array_plotter,
        )

    array_plotter.output.to_figure(structure=None, is_sub_plotter=False)

    plt.close()


def hyper_model_image(
    hyper_model_image,
    mask=None,
    positions=None,
    image_plane_pix_grid=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the image of a hyper_galaxies model image.

    Set *autolens.datas.arrays.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_model_image : datas.imaging.datas.Imaging
        The hyper_galaxies model image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotter.plot_array(
        array=hyper_model_image, mask=mask, grid=image_plane_pix_grid, points=positions
    )


def hyper_galaxy_image(
    hyper_galaxy_image,
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
        array=hyper_galaxy_image, mask=mask, grid=image_plane_pix_grid, points=positions
    )


def contribution_map(
    contribution_map,
    mask=None,
    positions=None,
    image_plane_pix_grid=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the image of a hyper_galaxies galaxy image.

    Set *autolens.datas.arrays.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    contribution_map : datas.imaging.datas.Imaging
        The hyper_galaxies galaxy image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotter.plot_array(
        array=contribution_map, mask=mask, grid=image_plane_pix_grid, points=positions
    )


def hyper_noise_map(
    hyper_noise_map,
    mask=None,
    positions=None,
    image_plane_pix_grid=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the image of a hyper_galaxies galaxy image.

    Set *autolens.datas.arrays.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_noise_map : datas.imaging.datas.Imaging
        The hyper_galaxies galaxy image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotter.plot_array(
        array=hyper_noise_map, mask=mask, grid=image_plane_pix_grid, points=positions
    )


def chi_squared_map(
    chi_squared_map,
    mask=None,
    positions=None,
    image_plane_pix_grid=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the image of a hyper_galaxies galaxy image.

    Set *autolens.datas.arrays.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    chi_squared_map : datas.imaging.datas.Imaging
        The hyper_galaxies galaxy image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotter.plot_array(
        array=chi_squared_map, mask=mask, grid=image_plane_pix_grid, points=positions
    )


def hyper_chi_squared_map(
    hyper_chi_squared_map,
    mask=None,
    positions=None,
    image_plane_pix_grid=None,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the image of a hyper_galaxies galaxy image.

    Set *autolens.datas.arrays.plotters.array_plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_chi_squared_map : datas.imaging.datas.Imaging
        The hyper_galaxies galaxy image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    array_plotter.plot_array(
        array=hyper_chi_squared_map,
        mask=mask,
        grid=image_plane_pix_grid,
        points=positions,
    )
