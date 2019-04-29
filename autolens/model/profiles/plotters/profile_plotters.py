from autolens.plotters import array_plotters
from autolens.plotters import plotter_util
from autolens.plotters import quantity_radii_plotters

def plot_intensities(
        light_profile, grid, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None, 
        as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Intensities', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
        output_path=None, output_format='show', output_filename='intensities'):
    """Plot the intensities (e.g. the image) of a light profile, on a regular grid of (y,x) coordinates.

    Set *autolens.hyper.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    light_profile : model.profiles.light_profiles.LightProfile
        The light profile whose intensities are plotted.
    grid : ndarray or hyper.array.grid_stacks.RegularGrid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
    """
    intensities = light_profile.intensities_from_grid(grid=grid)
    intensities = grid.scaled_array_2d_from_array_1d(intensities)

    array_plotters.plot_array(
        array=intensities, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_convergence(
        mass_profile, grid, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None, 
        as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Convergence', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
        output_path=None, output_format='show', output_filename='convergence'):
    """Plot the convergence of a mass profile, on a regular grid of (y,x) coordinates.

    Set *autolens.hyper.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose convergence is plotted.
    grid : ndarray or hyper.array.grid_stacks.RegularGrid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
    """
    convergence = mass_profile.convergence_from_grid(grid=grid)
    convergence = grid.scaled_array_2d_from_array_1d(convergence)

    array_plotters.plot_array(
        array=convergence, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_potential(
        mass_profile, grid, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None, 
        as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Potential', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
        output_path=None, output_format='show', output_filename='potential'):
    """Plot the potential of a mass profile, on a regular grid of (y,x) coordinates.

    Set *autolens.hyper.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose potential is plotted.
    grid : ndarray or hyper.array.grid_stacks.RegularGrid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
    """
    potential = mass_profile.potential_from_grid(grid=grid)
    potential = grid.scaled_array_2d_from_array_1d(potential)

    array_plotters.plot_array(
        array=potential, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_deflections_y(
        mass_profile, grid, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None, 
        as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Deflections (y)', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
        output_path=None, output_format='show', output_filename='deflections_y'):
    """Plot the y component of the deflection angles of a mass profile, on a regular grid of (y,x) coordinates.

    Set *autolens.hyper.array.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose y deflecton angles are plotted.
    grid : ndarray or hyper.array.grid_stacks.RegularGrid
        The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
    """
    deflections = mass_profile.deflections_from_grid(grid)
    deflections_y = grid.scaled_array_2d_from_array_1d(deflections[:, 0])

    array_plotters.plot_array(
        array=deflections_y, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)


def plot_deflections_x(
        mass_profile, grid, mask=None, extract_array_from_mask=False, zoom_around_mask=False, positions=None, 
        as_subplot=False,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), aspect='square',
        cmap='jet', norm='linear', norm_min=None, norm_max=None, linthresh=0.05, linscale=0.01,
        cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
        title='Deflections (x)', titlesize=16, xlabelsize=16, ylabelsize=16, xyticksize=16,
        mask_pointsize=10, position_pointsize=10.0, grid_pointsize=1,
        output_path=None, output_format='show', output_filename='deflections_x'):
    """Plot the x component of the deflection angles of a mass profile, on a regular grid of (y,x) coordinates.

     Set *autolens.hyper.array.plotters.array_plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
     mass_profile : model.profiles.mass_profiles.MassProfile
         The mass profile whose x deflecton angles are plotted.
     grid : ndarray or hyper.array.grid_stacks.RegularGrid
         The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2)
     """
    deflections = mass_profile.deflections_from_grid(grid)
    deflections_x = grid.scaled_array_2d_from_array_1d(deflections[:, 1])

    array_plotters.plot_array(
        array=deflections_x, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, positions=positions, as_subplot=as_subplot,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, aspect=aspect,
        cmap=cmap, norm=norm, norm_min=norm_min, norm_max=norm_max, linthresh=linthresh, linscale=linscale,
        cb_ticksize=cb_ticksize, cb_fraction=cb_fraction, cb_pad=cb_pad, 
        cb_tick_values=cb_tick_values, cb_tick_labels=cb_tick_labels,
        title=title, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        mask_pointsize=mask_pointsize, position_pointsize=position_pointsize, grid_pointsize=grid_pointsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)

def plot_luminosity_within_circle_in_electrons_per_second_as_function_of_radius(
        light_profile, minimum_radius=1.0e-4, maximum_radius=10.0, radii_bins=10,
        as_subplot=False, label='Light Profile', plot_axis_type='semilogy',
        effective_radius_line=None, einstein_radius_line=None,
        units='arcsec', kpc_per_arcsec=None, figsize=(7, 7), plot_legend=True,
        title='Luminosity (Electrons Per Second) vs Radius', ylabel='Luminosity (Electrons Per Second)', titlesize=16,
        xlabelsize=16, ylabelsize=16, xyticksize=16, legend_fontsize=12,
        output_path=None, output_format='show', output_filename='luminosity_vs_radius'):

    radii = plotter_util.quantity_radii_from_minimum_and_maximum_radii_and_radii_points(
        minimum_radius=minimum_radius, maximum_radius=maximum_radius, radii_points=radii_bins)

    luminosities = list(map(lambda radius :
                            light_profile.luminosity_within_circle_in_units(radius=radius),
                            radii))

    quantity_radii_plotters.plot_quantity_as_function_of_radius(
        quantity=luminosities, radii=radii,
        as_subplot=as_subplot, label=label, plot_axis_type=plot_axis_type,
        effective_radius_line=effective_radius_line, einstein_radius_line=einstein_radius_line,
        units=units, kpc_per_arcsec=kpc_per_arcsec, figsize=figsize, plot_legend=plot_legend,
        title=title, ylabel=ylabel, titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, xyticksize=xyticksize,
        legend_fontsize=legend_fontsize,
        output_path=output_path, output_format=output_format, output_filename=output_filename)