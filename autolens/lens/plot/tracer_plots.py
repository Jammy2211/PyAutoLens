import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

import autoarray as aa
import autogalaxy as ag

from autoarray.plot.array import plot_array
from autoarray.plot.utils import save_figure
from autoarray.plot.utils import numpy_lines as _to_lines, numpy_positions as _to_positions
from autogalaxy.plot.plot_utils import _critical_curves_from, _caustics_from


def subplot_tracer(
    tracer,
    grid: aa.type.Grid2DLike,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    use_log10: bool = False,
    positions=None,
):
    """Multi-panel subplot of the tracer: image, source images, and mass quantities.

    Panels (3x3 = 9 axes):
      0: full lensed image with critical curves
      1: source galaxy image (no caustics)
      2: source plane image (with caustics)
      3: lens galaxy image (log10)
      4: convergence (log10, with critical curves)
      5: potential (log10, with critical curves)
      6: deflections y (with critical curves)
      7: deflections x (with critical curves)
      8: magnification (with critical curves)
    """
    from autogalaxy.operate.lens_calc import LensCalc

    final_plane_index = len(tracer.planes) - 1
    traced_grids = tracer.traced_grid_2d_list_from(grid=grid)

    tan_cc, rad_cc = _critical_curves_from(tracer, grid)
    tan_ca, rad_ca = _caustics_from(tracer, grid)
    image_plane_lines = _to_lines(tan_cc, rad_cc)
    source_plane_lines = _to_lines(tan_ca, rad_ca)
    pos_list = _to_positions(positions)

    # --- compute arrays ---
    image = tracer.image_2d_from(grid=grid)

    source_galaxies = ag.Galaxies(galaxies=tracer.planes[final_plane_index])
    source_image = source_galaxies.image_2d_from(grid=traced_grids[final_plane_index])

    lens_galaxies = ag.Galaxies(galaxies=tracer.planes[0])
    lens_image = lens_galaxies.image_2d_from(grid=traced_grids[0])

    convergence = tracer.convergence_2d_from(grid=grid)
    potential = tracer.potential_2d_from(grid=grid)

    deflections = tracer.deflections_yx_2d_from(grid=grid)
    deflections_y = aa.Array2D(values=deflections.slim[:, 0], mask=grid.mask)
    deflections_x = aa.Array2D(values=deflections.slim[:, 1], mask=grid.mask)

    magnification = LensCalc.from_mass_obj(tracer).magnification_2d_from(grid=grid)

    fig, axes = plt.subplots(3, 3, figsize=(21, 21))
    axes_flat = list(axes.flatten())

    plot_array(array=image, ax=axes_flat[0], title="Image",
               lines=image_plane_lines, positions=pos_list, colormap=colormap,
               use_log10=use_log10)
    plot_array(array=source_image, ax=axes_flat[1], title="Source Image",
               colormap=colormap, use_log10=use_log10)
    plot_array(array=source_image, ax=axes_flat[2], title="Source Plane Image",
               lines=source_plane_lines, colormap=colormap, use_log10=use_log10)
    plot_array(array=lens_image, ax=axes_flat[3], title="Lens Image",
               colormap=colormap, use_log10=use_log10)
    plot_array(array=convergence, ax=axes_flat[4], title="Convergence",
               lines=image_plane_lines, colormap=colormap, use_log10=use_log10)
    plot_array(array=potential, ax=axes_flat[5], title="Potential",
               lines=image_plane_lines, colormap=colormap, use_log10=use_log10)
    plot_array(array=deflections_y, ax=axes_flat[6], title="Deflections Y",
               lines=image_plane_lines, colormap=colormap)
    plot_array(array=deflections_x, ax=axes_flat[7], title="Deflections X",
               lines=image_plane_lines, colormap=colormap)
    plot_array(array=magnification, ax=axes_flat[8], title="Magnification",
               lines=image_plane_lines, colormap=colormap)

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_tracer", format=output_format)


def subplot_lensed_images(
    tracer,
    grid: aa.type.Grid2DLike,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    use_log10: bool = False,
):
    """
    Produce a subplot with one panel per tracer plane showing each plane's image.

    For each plane in the tracer the galaxies in that plane are evaluated on
    the ray-traced grid for that plane, producing the lensed image
    contribution from those galaxies.  Each panel is titled
    ``"Image Of Plane <index>"``.

    Parameters
    ----------
    tracer : Tracer
        The tracer whose planes are ray-traced and imaged.
    grid : aa.type.Grid2DLike
        The 2-D (y, x) arc-second grid on which the lensed images are
        evaluated.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name.
    use_log10 : bool, optional
        If ``True`` the colour scale is applied on a log10 stretch.
    """
    traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
    n = tracer.total_planes

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    for plane_index in range(n):
        galaxies = ag.Galaxies(galaxies=tracer.planes[plane_index])
        image = galaxies.image_2d_from(grid=traced_grids[plane_index])
        plot_array(
            array=image,
            ax=axes_flat[plane_index],
            title=f"Image Of Plane {plane_index}",
            colormap=colormap,
            use_log10=use_log10,
        )

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_lensed_images", format=output_format)


def subplot_galaxies_images(
    tracer,
    grid: aa.type.Grid2DLike,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    use_log10: bool = False,
):
    """
    Produce a subplot showing per-galaxy images for every plane in the tracer.

    Renders the following panels in a single row:

    1. Lens-plane (plane 0) image.
    2. For each subsequent plane *i* (i ≥ 1):

       a. The lensed image of galaxies in plane *i* evaluated on the
          ray-traced grid (titled ``"Image Of Plane <i>"``).
       b. The source-plane image of galaxies in plane *i* (titled
          ``"Plane Image Of Plane <i>"``).

    The total number of panels is ``2 * total_planes - 1``.

    Parameters
    ----------
    tracer : Tracer
        The tracer whose planes are ray-traced and imaged.
    grid : aa.type.Grid2DLike
        The 2-D (y, x) arc-second grid on which the images are evaluated.
    output_path : str, optional
        Directory in which to save the figure.  If ``None`` the figure is
        not saved to disk.
    output_format : str, optional
        Image format passed to :func:`~autoarray.plot.utils.save_figure`.
    colormap : str, optional
        Matplotlib colormap name.
    use_log10 : bool, optional
        If ``True`` the colour scale is applied on a log10 stretch.
    """
    traced_grids = tracer.traced_grid_2d_list_from(grid=grid)
    n = 2 * tracer.total_planes - 1

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    idx = 0

    lens_galaxies = ag.Galaxies(galaxies=tracer.planes[0])
    lens_image = lens_galaxies.image_2d_from(grid=traced_grids[0])
    plot_array(
        array=lens_image,
        ax=axes_flat[idx],
        title="Image Of Plane 0",
        colormap=colormap,
        use_log10=use_log10,
    )
    idx += 1

    for plane_index in range(1, tracer.total_planes):
        plane_galaxies = ag.Galaxies(galaxies=tracer.planes[plane_index])
        plane_grid = traced_grids[plane_index]

        image = plane_galaxies.image_2d_from(grid=plane_grid)
        if idx < n:
            plot_array(
                array=image,
                ax=axes_flat[idx],
                title=f"Image Of Plane {plane_index}",
                colormap=colormap,
                use_log10=use_log10,
            )
            idx += 1

        if idx < n:
            plot_array(
                array=image,
                ax=axes_flat[idx],
                title=f"Plane Image Of Plane {plane_index}",
                colormap=colormap,
                use_log10=use_log10,
            )
            idx += 1

    plt.tight_layout()
    save_figure(fig, path=output_path, filename="subplot_galaxies_images", format=output_format)
