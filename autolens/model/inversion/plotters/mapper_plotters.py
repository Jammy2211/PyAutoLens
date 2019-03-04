import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from scipy.spatial import Voronoi

from autolens.data.array.plotters import plotter_util, grid_plotters
from autolens.model.inversion import mappers
from autolens.data.plotters import ccd_plotters


def plot_image_and_mapper(ccd_data, mapper, mask=None, positions=None, should_plot_centres=False, should_plot_grid=False,
                          should_plot_border=False,
                          image_pixels=None, source_pixels=None,
                          units='arcsec', kpc_per_arcsec=None,
                          output_path=None, output_filename='image_and_mapper', output_format='show'):

    rows, columns, figsize = plotter_util.get_subplot_rows_columns_figsize(number_subplots=2)
    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    ccd_plotters.plot_image(ccd_data=ccd_data, mask=mask, positions=positions, as_subplot=True,
                            units=units, kpc_per_arcsec=None, xyticksize=16,
                            norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                            figsize=None, aspect='equal', cmap='jet', cb_ticksize=10,
                            titlesize=10, xlabelsize=10, ylabelsize=10,
                            output_path=output_path, output_format=output_format)

    image_grid = convert_grid(grid=mapper.grid_stack.regular.unlensed_grid, units=units, kpc_per_arcsec=kpc_per_arcsec)

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    plot_image_plane_image_pixels(grid=image_grid, image_pixels=image_pixels, point_colors=point_colors)
    plot_image_plane_source_pixels(grid=image_grid, mapper=mapper, source_pixels=source_pixels,
                                   point_colors=point_colors)

    plt.subplot(rows, columns, 2)

    plot_mapper(mapper=mapper, should_plot_centres=should_plot_centres, should_plot_grid=should_plot_grid,
                should_plot_border=should_plot_border,
                image_pixels=image_pixels, source_pixels=source_pixels,
                as_subplot=True,
                units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=None)

    plotter_util.output_subplot_array(output_path=output_path, output_filename=output_filename, output_format=output_format)
    plt.close()

def plot_mapper(mapper, should_plot_centres=False, should_plot_grid=False, should_plot_border=False,
                image_pixels=None, source_pixels=None, as_subplot=False,
                units='arcsec', kpc_per_arcsec=None,
                xyticksize=16, figsize=(7, 7),
                title='Mapper', titlesize=16, xlabelsize=16, ylabelsize=16,
                output_path=None, output_filename='mapper', output_format='show'):

    if isinstance(mapper, mappers.RectangularMapper):

        plot_rectangular_mapper(mapper=mapper, should_plot_centres=should_plot_centres,
                                should_plot_grid=should_plot_grid, should_plot_border=should_plot_border,
                                image_pixels=image_pixels, source_pixels=source_pixels, as_subplot=as_subplot,
                                units=units, kpc_per_arcsec=kpc_per_arcsec,
                                xyticksize=xyticksize, figsize=figsize,
                                title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_rectangular_mapper(mapper, should_plot_centres=False, should_plot_grid=False, should_plot_border=False,
                            image_pixels=None, source_pixels=None, as_subplot=False,
                            units='arcsec', kpc_per_arcsec=None,
                            xyticksize=16, figsize=(7, 7),
                            title='Rectangular Mapper', titlesize=16, xlabelsize=16, ylabelsize=16,
                            output_path=None, output_filename='rectangular_mapper', output_format='show'):

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    set_axis_limits(mapper=mapper, units=units, kpc_per_arcsec=kpc_per_arcsec)
    plot_rectangular_pixelization_lines(mapper=mapper, units=units, kpc_per_arcsec=kpc_per_arcsec)

    plotter_util.set_title(title=title, titlesize=titlesize)
    grid_plotters.set_xy_labels(units=units, kpc_per_arcsec=kpc_per_arcsec,
                                xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)

    plot_centres(should_plot_centres=should_plot_centres, mapper=mapper, units=units, kpc_per_arcsec=kpc_per_arcsec)

    plot_plane_grid(should_plot_grid=should_plot_grid, mapper=mapper, as_subplot=True, units=units,
                    kpc_per_arcsec=kpc_per_arcsec, pointsize=10, xyticksize=xyticksize,
                    title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize)

    plot_border(should_plot_border=should_plot_border, mapper=mapper, as_subplot=True, units=units,
                kpc_per_arcsec=kpc_per_arcsec, pointsize=30, xyticksize=xyticksize,
                title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize)

    mapper_grid = convert_grid(grid=mapper.grid_stack.regular, units=units, kpc_per_arcsec=kpc_per_arcsec)

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    plot_source_plane_image_pixels(grid=mapper_grid, image_pixels=image_pixels, point_colors=point_colors)
    plot_source_plane_source_pixels(grid=mapper_grid, mapper=mapper, source_pixels=source_pixels,
                                    point_colors=point_colors)

    plotter_util.output_figure(None, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename,
                               output_format=output_format)
    plotter_util.close_figure(as_subplot=as_subplot)

def plot_voronoi_mapper(mapper, solution_vector, should_plot_centres=True, should_plot_grid=True, should_plot_border=False,
                        image_pixels=None, source_pixels=None, as_subplot=False,
                        units='arcsec', kpc_per_arcsec=None,
                        xyticksize=16, figsize=(7, 7),
                        title='Rectangular Mapper', titlesize=16, xlabelsize=16, ylabelsize=16,
                        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
                        output_path=None, output_filename='voronoi_mapper', output_format='show'):

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)
 #   plt.figaspect(1)

    set_axis_limits(mapper=mapper, units=units, kpc_per_arcsec=kpc_per_arcsec)

    regions_SP, vertices_SP = voronoi_finite_polygons_2d(mapper.voronoi)

    color_values = solution_vector[:] / np.max(solution_vector)
    cmap = plt.get_cmap('jet')

    set_colorbar(cmap=cmap, color_values=color_values, cb_fraction=cb_fraction, cb_pad=cb_pad,
                 cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels)

    for region, index in zip(regions_SP, range(mapper.pixels)):
        polygon = vertices_SP[region]
        col = cmap(color_values[index])
        plt.fill(*zip(*polygon), alpha=0.7, facecolor=col, lw=0.0)

    plotter_util.set_title(title=title, titlesize=titlesize)
    grid_plotters.set_xy_labels(units=units, kpc_per_arcsec=kpc_per_arcsec,
                                xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize)

    plot_centres(should_plot_centres=should_plot_centres, mapper=mapper, units=units, kpc_per_arcsec=kpc_per_arcsec)

    plot_plane_grid(should_plot_grid=should_plot_grid, mapper=mapper, as_subplot=True, units=units,
                    kpc_per_arcsec=kpc_per_arcsec, pointsize=10, xyticksize=xyticksize,
                    title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize)

    plot_border(should_plot_border=should_plot_border, mapper=mapper, as_subplot=True, units=units,
                kpc_per_arcsec=kpc_per_arcsec, pointsize=30, xyticksize=xyticksize,
                title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize)

    mapper_grid = convert_grid(grid=mapper.grid_stack.regular, units=units, kpc_per_arcsec=kpc_per_arcsec)

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    plot_source_plane_image_pixels(grid=mapper_grid, image_pixels=image_pixels, point_colors=point_colors)
    plot_source_plane_source_pixels(grid=mapper_grid, mapper=mapper, source_pixels=source_pixels,
                                    point_colors=point_colors)

    plotter_util.output_figure(None, as_subplot=as_subplot, output_path=output_path, output_filename=output_filename,
                               output_format=output_format)
    plotter_util.close_figure(as_subplot=as_subplot)

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def plot_rectangular_pixelization_lines(mapper, units, kpc_per_arcsec):

    if units in 'arcsec' or kpc_per_arcsec is None:

        ys = np.linspace(mapper.geometry.arc_second_minima[0], mapper.geometry.arc_second_maxima[0], mapper.shape[0]+1)
        xs = np.linspace(mapper.geometry.arc_second_minima[1], mapper.geometry.arc_second_maxima[1], mapper.shape[1]+1)

    elif units in 'kpc':

        ys = np.linspace(mapper.geometry.arc_second_minima[0]*kpc_per_arcsec,
                         mapper.geometry.arc_second_maxima[0]*kpc_per_arcsec, mapper.shape[0]+1)
        xs = np.linspace(mapper.geometry.arc_second_minima[1]*kpc_per_arcsec,
                         mapper.geometry.arc_second_maxima[1]*kpc_per_arcsec, mapper.shape[1]+1)

    # grid lines
    for x in xs:
        plt.plot([x, x], [ys[0], ys[-1]], color='black', linestyle='-')
    for y in ys:
        plt.plot([xs[0], xs[-1]], [y, y], color='black', linestyle='-')

def set_axis_limits(mapper, units, kpc_per_arcsec):

    if units in 'arcsec' or kpc_per_arcsec is None:

        grid_plotters.set_axis_limits(axis_limits=np.asarray([mapper.geometry.arc_second_minima[1],
                                                              mapper.geometry.arc_second_maxima[1],
                                                              mapper.geometry.arc_second_minima[0],
                                                              mapper.geometry.arc_second_maxima[0]]))

    elif units in 'kpc':

        grid_plotters.set_axis_limits(axis_limits=np.asarray([mapper.geometry.arc_second_minima[1] * kpc_per_arcsec,
                                                              mapper.geometry.arc_second_maxima[1] * kpc_per_arcsec,
                                                              mapper.geometry.arc_second_minima[0] * kpc_per_arcsec,
                                                              mapper.geometry.arc_second_maxima[0] * kpc_per_arcsec]))

def set_colorbar(cmap, color_values, cb_fraction, cb_pad, cb_tick_values, cb_tick_labels):

    cax = cm.ScalarMappable(cmap=cmap)
    cax.set_array(color_values)

    if cb_tick_values is None and cb_tick_labels is None:
        plt.colorbar(mappable=cax, fraction=cb_fraction, pad=cb_pad)
    elif cb_tick_values is not None and cb_tick_labels is not None:
        cb = plt.colorbar(mappable=cax, fraction=cb_fraction, pad=cb_pad, ticks=cb_tick_values)
        cb.ax.set_yticklabels(cb_tick_labels)

def plot_centres(should_plot_centres, mapper, units, kpc_per_arcsec):

    if should_plot_centres:

        if units in 'arcsec' or kpc_per_arcsec is None:

            pixel_centres = mapper.geometry.pixel_centres

        elif units in 'kpc':

            pixel_centres = mapper.geometry.pixel_centres * kpc_per_arcsec

        plt.scatter(y=pixel_centres[:,0], x=pixel_centres[:,1], s=3, c='r')

def plot_plane_grid(should_plot_grid, mapper, as_subplot, units, kpc_per_arcsec, pointsize, xyticksize, title,
                    titlesize, xlabelsize, ylabelsize):

    if should_plot_grid:

        grid_units = convert_grid(grid=mapper.grid_stack.regular, units=units, kpc_per_arcsec=kpc_per_arcsec)

        grid_plotters.plot_grid(grid=grid_units, as_subplot=as_subplot, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                pointsize=pointsize, xyticksize=xyticksize, title=title, titlesize=titlesize,
                                xlabelsize=xlabelsize, ylabelsize=ylabelsize)

def plot_border(should_plot_border, mapper, as_subplot, units, kpc_per_arcsec, pointsize, xyticksize, title,
                titlesize, xlabelsize, ylabelsize):

    if should_plot_border:

        border_arcsec = mapper.grid_stack.regular[mapper.border]
        border_units = convert_grid(grid=border_arcsec, units=units, kpc_per_arcsec=kpc_per_arcsec)

        grid_plotters.plot_grid(grid=border_units, as_subplot=as_subplot, units=units, kpc_per_arcsec=kpc_per_arcsec,
                                pointsize=pointsize, pointcolor='y', xyticksize=xyticksize, title=title,
                                titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize)

def plot_image_plane_image_pixels(grid, image_pixels, point_colors):

    if image_pixels is not None:

        for image_pixel_set in image_pixels:
            color = next(point_colors)
            plt.scatter(y=np.asarray(grid[image_pixel_set, 0]),
                        x=np.asarray(grid[image_pixel_set, 1]), color=color, s=10.0)

def plot_image_plane_source_pixels(grid, mapper, source_pixels, point_colors):

    if source_pixels is not None:

        for source_pixel_set in source_pixels:
            color = next(point_colors)
            for source_pixel in source_pixel_set:
                plt.scatter(y=np.asarray(grid[mapper.pix_to_regular[source_pixel], 0]),
                            x=np.asarray(grid[mapper.pix_to_regular[source_pixel], 1]), s=8, color=color)

def plot_source_plane_image_pixels(grid, image_pixels, point_colors):

    if image_pixels is not None:

        for image_pixel_set in image_pixels:
            color = next(point_colors)
            plt.scatter(y=np.asarray(grid[[image_pixel_set],0]),
                        x=np.asarray(grid[[image_pixel_set],1]), s=8, color=color)

def plot_source_plane_source_pixels(grid, mapper, source_pixels, point_colors):

    if source_pixels is not None:

        for source_pixel_set in source_pixels:
            color = next(point_colors)
            for source_pixel in source_pixel_set:
                plt.scatter(y=np.asarray(grid[mapper.pix_to_regular[source_pixel], 0]),
                            x=np.asarray(grid[mapper.pix_to_regular[source_pixel], 1]), s=8, color=color)

def convert_grid(grid, units, kpc_per_arcsec):

    if units in 'arcsec' or kpc_per_arcsec is None:
        return grid
    elif units in 'kpc':
        return grid * kpc_per_arcsec