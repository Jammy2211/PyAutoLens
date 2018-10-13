from matplotlib import pyplot as plt

from autolens import conf
from autolens.plotting import plotters
from autolens.plotting import plotter_tools

def plot_intensities(light_profile, grid, as_subplot=False,
                     units='arcsec', kpc_per_arcsec=None,
                     xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                     figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                     title='Intensities', titlesize=46, xlabelsize=36, ylabelsize=36,
                     output_path=None, output_format='show', output_filename='intensities'):

    intensities = light_profile.intensities_from_grid(grid=grid)
    intensities = grid.scaled_array_from_array_1d(intensities)

    plotters.plot_intensities(intensities, units, as_subplot,
                              kpc_per_arcsec, xyticksize, norm, norm_min, norm_max, linthresh,
                              linscale, figsize, aspect, cmap, cb_ticksize, title, titlesize, xlabelsize, ylabelsize,
                              output_path, output_format, output_filename)

def plot_surface_density(mass_profile, grid, as_subplot=False,
                         units='arcsec', kpc_per_arcsec=None,
                         xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                         figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                         title='Surface Density', titlesize=46, xlabelsize=36, ylabelsize=36,
                         output_path=None, output_format='show', output_filename='surface_density'):

    surface_density = mass_profile.surface_density_from_grid(grid=grid)
    surface_density = grid.scaled_array_from_array_1d(surface_density)

    plotters.plot_surface_density(surface_density, units, as_subplot,
                                  kpc_per_arcsec, xyticksize, norm, norm_min, norm_max, linthresh,
                                  linscale, figsize, aspect, cmap, cb_ticksize, title, titlesize, xlabelsize, ylabelsize,
                                  output_path, output_format, output_filename)

def plot_potential(mass_profile, grid, as_subplot=False,
                   units='arcsec', kpc_per_arcsec=None,
                   xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                   figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                   title='Potential', titlesize=46, xlabelsize=36, ylabelsize=36,
                   output_path=None, output_format='show', output_filename='potential'):

    potential = mass_profile.potential_from_grid(grid=grid)
    potential = grid.scaled_array_from_array_1d(potential)

    plotters.plot_potential(potential, units, as_subplot,
                            kpc_per_arcsec, xyticksize, norm, norm_min, norm_max, linthresh,
                            linscale, figsize, aspect, cmap, cb_ticksize, title, titlesize, xlabelsize, ylabelsize,
                            output_path, output_format, output_filename)


def plot_deflections_y(mass_profile, grid, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
                       figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                       title='Deflections (y)', titlesize=46, xlabelsize=36, ylabelsize=36,
                       output_path=None, output_format='show', output_filename='deflections_y'):
    
    deflections = mass_profile.deflections_from_grid(grid)
    deflections_y = grid.scaled_array_from_array_1d(deflections[:,0])

    plotters.plot_deflections_y(deflections_y, units, as_subplot,
                                kpc_per_arcsec, xyticksize, norm, norm_min, norm_max, linthresh,
                                linscale, figsize, aspect, cmap, cb_ticksize, title, titlesize, xlabelsize, ylabelsize,
                                output_path, output_format, output_filename)


def plot_deflections_x(mass_profile, grid, as_subplot=False,
                       units='arcsec', kpc_per_arcsec=None,
                       xyticksize=40, norm='linear', norm_min=None, norm_max=None, linthresh=0.05,
                       linscale=0.01,
                       figsize=(20, 15), aspect='auto', cmap='jet', cb_ticksize=20,
                       title='Deflections (x)', titlesize=46, xlabelsize=36, ylabelsize=36,
                       output_path=None, output_format='show', output_filename='deflections_x'):

    deflections = mass_profile.deflections_from_grid(grid)
    deflections_x = grid.scaled_array_from_array_1d(deflections[:, 1])

    plotters.plot_deflections_x(deflections_x, units, as_subplot,
                                kpc_per_arcsec, xyticksize, norm, norm_min, norm_max, linthresh,
                                linscale, figsize, aspect, cmap, cb_ticksize, title, titlesize, xlabelsize, ylabelsize,
                                output_path, output_format, output_filename)