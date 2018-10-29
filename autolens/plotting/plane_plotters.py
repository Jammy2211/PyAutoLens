from matplotlib import pyplot as plt

from autolens.plotting import tools
from autolens.plotting import tools_array
from autolens.plotting import tools_grid

def plot_image_plane_image(plane, image_index=0, mask=None, positions=None, grid=None, as_subplot=False,
                           units='arcsec', figsize=(7, 7), aspect='equal',
                           cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                           cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                           title='Plane Image-Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                           mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
                           output_path=None, output_format='show', output_filename='plane_image_plane_image'):

    tools_array.plot_array(array=plane.image_plane_images[image_index], mask=mask, positions=positions, grid=grid,
                           as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=plane.kpc_per_arcsec_proper, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           mask_pointsize=mask_pointsize, position_pointsize=position_pointsize,
                           grid_pointsize=grid_pointsize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_plane_image(plane, image_index=0, positions=None, plot_grid=False, as_subplot=False,
                     units='arcsec', figsize=(7, 7), aspect='equal',
                     cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                     title='Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                     position_pointsize=10, grid_pointsize=1,
                     output_path=None, output_format='show', output_filename='plane_image'):

    if plot_grid:
        grid = plane.grids[image_index].image
    else:
        grid = None

    tools_array.plot_array(array=plane.plane_images[image_index], positions=positions, grid=grid, as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=plane.kpc_per_arcsec_proper, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_surface_density(plane, as_subplot=False,
                         units='arcsec', figsize=(7, 7), aspect='equal',
                         cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                         title='Plane Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                         output_path=None, output_format='show', output_filename='plane_surface_density'):

    tools_array.plot_array(array=plane.surface_density, as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=plane.kpc_per_arcsec_proper, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_potential(plane, as_subplot=False,
                   units='arcsec', figsize=(7, 7), aspect='equal',
                   cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                   title='Plane Potential', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                   output_path=None, output_format='show', output_filename='plane_potential'):

    tools_array.plot_array(array=plane.potential, as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=plane.kpc_per_arcsec_proper, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_deflections_y(plane, as_subplot=False,
                       units='arcsec', figsize=(7, 7), aspect='equal',
                       cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Plane Deflections (y)', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                       output_path=None, output_format='show', output_filename='plane_deflections_y'):

    tools_array.plot_array(array=plane.deflections_y, as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=plane.kpc_per_arcsec_proper, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_deflections_x(plane, as_subplot=False,
                       units='arcsec', figsize=(7, 7), aspect='equal',
                       cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01,
                       title='Plane Deflections (x)', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                       output_path=None, output_format='show', output_filename='plane_deflections_x'):

    tools_array.plot_array(array=plane.deflections_x, as_subplot=as_subplot,
                           units=units, kpc_per_arcsec=plane.kpc_per_arcsec_proper, figsize=figsize, aspect=aspect,
                           cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max,
                           linthresh=linthresh, linscale=linscale,
                           cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad,
                           title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                           xyticksize=xyticksize,
                           output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_image_and_source_plane_subplot(image_plane, source_plane, points=None, axis_limits=None,
                    units='arcsec',
                    output_path=None, output_format='show', output_filename='image_and_source_plane_grids'):

    rows, columns, figsize = tools.get_subplot_rows_columns_figsize(number_subplots=2)

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plot_plane_grid(plane=image_plane, axis_limits=axis_limits, points=points, as_subplot=True,
                    units=units,
                    pointsize=3, xyticksize=16, titlesize=10, xlabelsize=10, ylabelsize=10,
                    title='Image-plane Grid',
                    output_path=output_path, output_filename=output_filename, output_format=output_format)

    plt.subplot(rows, columns, 2)

    plot_plane_grid(plane=source_plane, axis_limits=axis_limits, points=points, as_subplot=True,
                    units=units,
                    pointsize=3, xyticksize=16, titlesize=10, xlabelsize=10, ylabelsize=10,
                    title='Source-plane Grid',
                    output_path=output_path, output_filename=output_filename, output_format=output_format)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename, output_format=output_format)
    plt.close()


def plot_plane_grid(plane, axis_limits=None, points=None, as_subplot=False,
                    units='arcsec',
                    figsize=(12, 8), pointsize=3,
                    title='Plane Grid', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
                    output_path=None, output_format='show', output_filename='plane_grid'):

    tools_grid.plot_grid(grid=plane.grids[0].image, points=points, axis_limits=axis_limits, as_subplot=as_subplot,
                         units=units, kpc_per_arcsec=plane.kpc_per_arcsec_proper,
                         figsize=figsize, pointsize=pointsize, xyticksize=xyticksize,
                         title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                         output_path=output_path, output_format=output_format, output_filename=output_filename)