from matplotlib import pyplot as plt

from autolens.plotting import plotters

def plot_image_plane_image(plane, mask=None, positions=None, grid=None, as_subplot=False,
                           units='arcsec', kpc_per_arcsec=None,
                           xyticksize=16, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                           figsize=(7, 7), aspect='equal', cmap='jet', cb_ticksize=16,
                           title='Plane Image-Plane Image', titlesize=16, xlabelsize=16, ylabelsize=16,
                           output_path=None, output_format='show', output_filename='plane_image_plane_image'):

    plotters.plot_image_plane_image(image_plane_image=plane.image_plane_image, mask=mask, positions=positions,
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

    plotters.plot_plane_image(plane_image=plane.plane_image, positions=positions, plot_grid=plot_grid,
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

    plotters.plot_surface_density(surface_density=plane.surface_density, as_subplot=as_subplot,
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

    plotters.plot_potential(potential=plane.potential, as_subplot=as_subplot,
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

    plotters.plot_deflections_y(deflections_y=plane.deflections_y, as_subplot=as_subplot,
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

    plotters.plot_deflections_x(deflections_x=plane.deflections_x, as_subplot=as_subplot,
                              units=units, kpc_per_arcsec=kpc_per_arcsec, xyticksize=xyticksize,
                              norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh,
                              linscale=linscale, figsize=figsize, aspect=aspect, cmap=cmap, cb_ticksize=cb_ticksize,
                              title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                              output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_plane_grid(plane, xmin=None, xmax=None, ymin=None, ymax=None,
                    figsize=(12, 8), pointsize=3, xyticksize=16,
                    title='Plane Grid', titlesize=16, xlabelsize=16, ylabelsize=16,
                    output_path=None, output_format='show', output_filename='plane_grid'):

    plotters.plot_grid(grid=plane.grids.image, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                       figsize=figsize, pointsize=pointsize, xyticksize=xyticksize,
                       title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize,
                       output_path=output_path, output_format=output_format, output_filename=output_filename)