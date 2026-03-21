import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

import autoarray as aa
import autogalaxy as ag

from autolens.plot.plot_utils import (
    plot_array,
    _to_lines,
    _to_positions,
    _save_subplot,
    _critical_curves_from,
    _caustics_from,
)


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
    _save_subplot(fig, output_path, "subplot_tracer", output_format)


def subplot_lensed_images(
    tracer,
    grid: aa.type.Grid2DLike,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    use_log10: bool = False,
):
    """One panel per plane showing the image of the galaxies in that plane."""
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
    _save_subplot(fig, output_path, "subplot_lensed_images", output_format)


def subplot_galaxies_images(
    tracer,
    grid: aa.type.Grid2DLike,
    output_path: Optional[str] = None,
    output_format: str = "png",
    colormap: str = "jet",
    use_log10: bool = False,
):
    """Plane 0 image + for each plane > 0: lensed image + source plane image."""
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
    _save_subplot(fig, output_path, "subplot_galaxies_images", output_format)
