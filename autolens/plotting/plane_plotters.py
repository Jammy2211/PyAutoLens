from matplotlib import pyplot as plt

from autolens.plotting import tools
from autolens.plotting import plot_array
from autolens.plotting import plot_grid

def plot_image_plane_image(plane, mask=None, positions=None, grid=None, as_subplot=False,
                           units='arcsec', kpc_per_arcsec=None,
                           xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                           figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                           title='Plane Image-Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                           output_path=None, output_format='show', output_filename='plane_image_plane_image'):

    plot_array.plot_image_plane_image(image_plane_image=plane.image_plane_image, mask=mask, positions=positions,
                                      grid=grid, as_subplot=as_subplot,
                                      units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                      norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                      linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                                      title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                      output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_plane_image(plane, positions=None, plot_grid=False, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                     title='Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                     output_path=None, output_format='show', output_filename='plane_image'):

    plot_array.plot_plane_image(plane_image=plane.plane_image, positions=positions, plot_grid=plot_grid,
                                as_subplot=as_subplot,
                                units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                                title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_surface_density(plane, as_subplot=False,
                         units='arcsec', kpc_per_arcsec=None,
                         xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                         title='Plane Surface Density', titlesize=16, xlabelsize=16, ylabelsize=16,
                         output_path=None, output_format='show', output_filename='plane_surface_density'):

    plot_array.plot_surface_density(surface_density=plane.surface_density, as_subplot=as_subplot,
                                    units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                    norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                    linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                                    title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                    output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_potential(plane, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                   title='Plane Potential', titlesize=16, xlabelsize=16, ylabelsize=16,
                   output_path=None, output_format='show', output_filename='plane_potential'):

    plot_array.plot_potential(potential=plane.potential, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)
    
def plot_deflections_y(plane, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                       title='Plane Deflections (y)', titlesize=16, xlabelsize=16, ylabelsize=16,
                       output_path=None, output_format='show', output_filename='plane_deflections_y'):

    plot_array.plot_deflections_y(deflections_y=plane.deflections_y, as_subplot=as_subplot,
                                  units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                  norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                  linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                                  title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                  output_path=output_path, output_format=output_format, output_filename=output_filename)
    
def plot_deflections_x(plane, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                       title='Plane Deflections (x)', titlesize=16, xlabelsize=16, ylabelsize=16,
                       output_path=None, output_format='show', output_filename='plane_deflections_x'):

    plot_array.plot_deflections_x(deflections_x=plane.deflections_x, as_subplot=as_subplot,
                                  units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                                  norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                                  linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                                  title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                                  output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_image_and_source_plane_subplot(image_plane, source_plane, points=None, axis_limits=None,
                    units='arcsec', kpc_per_arcsec=None,
                    output_path=None, output_format='show', output_filename='image_and_source_plane_grids'):

    rows, columns, figsize = tools.get_subplot_rows_columns_figsize(number_subplots=2)

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    plot_plane_grid(plane=image_plane, axis_limits=axis_limits, points=points, as_subplot=True,
                    units=units, kpc_per_arcsec=kpc_per_arcsec,
                    pointsize=3, xyticksize=16, titlesize=10, xlabelsize=10, ylabelsize=10,
                    title='Image-plane Grid',
                    output_path=output_path, output_filename=output_filename, output_format=output_format)

    plt.subplot(rows, columns, 2)

    plot_plane_grid(plane=source_plane, axis_limits=axis_limits, points=points, as_subplot=True,
                    units=units, kpc_per_arcsec=kpc_per_arcsec,
                    pointsize=3, xyticksize=16, titlesize=10, xlabelsize=10, ylabelsize=10,
                    title='Source-plane Grid',
                    output_path=output_path, output_filename=output_filename, output_format=output_format)

    tools.output_subplot_array(output_path=output_path, output_filename=output_filename, output_format=output_format)
    plt.close()


def plot_plane_grid(plane, axis_limits=None, points=None, as_subplot=False,
                    units='arcsec', kpc_per_arcsec=None,
                    figsize=(12, 8), pointsize=3, xyticksize=16,
                    title='Plane Grid', titlesize=16, xlabelsize=16, ylabelsize=16,
                    output_path=None, output_format='show', output_filename='plane_grid'):

    plot_grid.plot_grid(grid=plane.grids.image, points=points, axis_limits=axis_limits, as_subplot=as_subplot,
                         units=units, kpc_per_arcsec=kpc_per_arcsec,
                         figsize=figsize, pointsize=pointsize, xyticksize=xyticksize,
                         title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                         output_path=output_path, output_format=output_format, output_filename=output_filename)