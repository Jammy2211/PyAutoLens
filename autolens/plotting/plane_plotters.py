from matplotlib import pyplot as plt

from autolens.plotting import plotters

def plot_image_plane_image(plane, mask=None, positions=None, grid=None, as_subplot=False,
                           units='arcsec', kpc_per_arcsec=None,
                           xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                           figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                           title='Image-Plane Image', titlesize=46, xlabelsize=36, ylabelsize=36,
                           output_path=None, output_format='show', output_filename='plane_image_plane_image'):

    plotters.plot_image_plane_image(plane.image_plane_image, mask, positions, grid, as_subplot,
                           units, kpc_per_arcsec, xyticksize, norm, norm_min,
                           norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                           titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)

def plot_plane_image(plane, positions=None, plot_grid=False, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                     title='Plane Image', titlesize=46, xlabelsize=36, ylabelsize=36,
                     output_path=None, output_format='show', output_filename='plane_image'):

    plotters.plot_plane_image(plane.image_plane_image, positions, plot_grid, as_subplot,
                     units, kpc_per_arcsec, xyticksize, norm, norm_min,
                     norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                     titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)

def plot_surface_density(plane, as_subplot=False,
                         units='arcsec', kpc_per_arcsec=None,
                         xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                         title='Plane Surface Density', titlesize=46, xlabelsize=36, ylabelsize=36,
                         output_path=None, output_format='show', output_filename='plane_surface_density'):

    plotters.plot_surface_density(plane.surface_density, as_subplot,
                                  units, kpc_per_arcsec, xyticksize, norm, norm_min,
                                  norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                                  titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)

def plot_potential(plane, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                   title='Plane Surface Density', titlesize=46, xlabelsize=36, ylabelsize=36,
                   output_path=None, output_format='show', output_filename='plane_potential'):

    plotters.plot_potential(plane.potential, as_subplot,
                            units, kpc_per_arcsec, xyticksize, norm, norm_min,
                            norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                            titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)
    
def plot_deflections_y(plane, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                       title='Plane Surface Density', titlesize=46, xlabelsize=36, ylabelsize=36,
                       output_path=None, output_format='show', output_filename='plane_deflections_y'):

    plotters.plot_deflections_y(plane.deflections_y, as_subplot,
                                units, kpc_per_arcsec, xyticksize, norm, norm_min,
                                norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                                titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)
    
def plot_deflections_x(plane, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                       title='Plane Surface Density', titlesize=46, xlabelsize=36, ylabelsize=36,
                       output_path=None, output_format='show', output_filename='plane_deflections_x'):

    plotters.plot_deflections_x(plane.deflections_x, as_subplot,
                                units, kpc_per_arcsec, xyticksize, norm, norm_min,
                                norm_max, linthresh, linscale, figsize, aspect, cmap, cb_ticksize, title,
                                titlesize, xlabelsize, ylabelsize, output_path, output_format, output_filename)

def plot_plane_grid(plane, xmin=None, xmax=None, ymin=None, ymax=None,
                    output_path=None, output_format='show', output_filename='plane_grid'):

    plotters.plot_grid(grid=plane.grids.image, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, output_path=output_path,
                       output_format=output_format, output_filename=output_filename)