from matplotlib import pyplot as plt

from autolens import conf
from autolens.plotting import array_plotters

def plot_intensities(light_profile, grid, output_path=None, output_filename='intensities', output_format='show'):

    intensities = light_profile.intensities_from_grid(grid=grid)
    intensities = grid.scaled_array_from_array_1d(intensities)

    array_plotters.plot_array(
        array=intensities, points=None, grid=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None,
        xticks=intensities.xticks, yticks=intensities.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Intensities', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_surface_density(mass_profile, grid, output_path=None, output_filename='surface_density', output_format='show'):

    surface_density = mass_profile.surface_density_from_grid(grid=grid)
    surface_density = grid.scaled_array_from_array_1d(surface_density)

    array_plotters.plot_array(
        array=surface_density, points=None, grid=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None,
        xticks=surface_density.xticks, yticks=surface_density.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Surface density', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=output_filename, output_format=output_format)
    
def plot_potential(mass_profile, grid, output_path=None, output_filename='potential', output_format='show'):

    potential = mass_profile.potential_from_grid(grid=grid)
    potential = grid.scaled_array_from_array_1d(potential)

    array_plotters.plot_array(
        array=potential, points=None, grid=None, as_subplot=False,
        units='arcsec', kpc_per_arcsec=None,
        xticks=potential.xticks, yticks=potential.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Potential', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=output_filename, output_format=output_format)

def plot_deflections(mass_profile, grid, output_path=None, output_filename='deflections',
                     output_format='show'):

    deflections = mass_profile.deflections_from_grid(grid)

    deflections_y = grid.scaled_array_from_array_1d(deflections[:,0])
    deflections_x = grid.scaled_array_from_array_1d(deflections[:,1])

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)

    array_plotters.plot_array(
        array=deflections_x, points=None, grid=None, as_subplot=True,
        units='arcsec', kpc_per_arcsec=None,
        xticks=deflections_x.xticks, yticks=deflections_x.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Deflection angles (x)', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    plt.subplot(1, 2, 2)

    array_plotters.plot_array(
        array=deflections_y, points=None, grid=None, as_subplot=True,
        units='arcsec', kpc_per_arcsec=None,
        xticks=deflections_y.xticks, yticks=deflections_y.yticks, xyticksize=16,
        norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        figsize=None, aspect='auto', cmap='jet', cb_ticksize=16,
        title='Deflection angles (y)', titlesize=16, xlabelsize=16, ylabelsize=16,
        output_path=output_path, output_filename=None, output_format=output_format)

    array_plotters.output_subplot_array(output_path=output_path, output_filename=output_filename,
                                        output_format=output_format)
    plt.close()