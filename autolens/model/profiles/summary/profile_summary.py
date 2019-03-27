from autolens.model.profiles import mass_profiles as mp

def summarize_mass_profile(summary_file, mass_profile, critical_surface_mass_density,
                           cosmic_average_mass_density_arcsec, radii):

    summary_file.write('Mass Profile = ' + mass_profile.__class__.__name__ + '\n\n')
    summarize_einstein_radius_and_mass(summary_file=summary_file, mass_profile=mass_profile,
                                       critical_surface_mass_density=critical_surface_mass_density)
    summarize_mass_within_radii(summary_file=summary_file, mass_profile=mass_profile,
                                critical_surface_mass_density=critical_surface_mass_density,
                                radii=radii)

    if isinstance(mass_profile, mp.SphericalTruncatedNFWChallenge):
        summarize_truncated_nfw_challenge_mass_profile(summary_file=summary_file, truncated_nfw_challenge=mass_profile)



def summarize_einstein_radius_and_mass(summary_file, mass_profile, critical_surface_mass_density):
    """ Summarize the mass at the Einstein radius of the mass profile.

    Parameters
    -----------
    summary_file : file
        The summary file the mass-profiles information is written to.
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile instance whose mass within radii are summarized.
    critical_surface_mass_density : float
        The critical surface mass density of the lensing system, required to convert the mass to physical units.
    """

    einstein_mass = mass_profile.mass_within_circle_in_mass_units(
        radius=mass_profile.einstein_radius, critical_surface_mass_density=critical_surface_mass_density)

    summary_file.write('Einstein Radius = {:.2f}"\n'.format(mass_profile.einstein_radius))
    summary_file.write('Mass within Einstein Radius = {:.4e} solMass\n'.format(einstein_mass))

def summarize_mass_within_radii(summary_file, mass_profile, critical_surface_mass_density, radii):
    """ Summarize the mass within a set of input radii of a given mass profile.

    Parameters
    -----------
    summary_file : file
        The summary file the mass-profiles information is written to.
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile instance whose mass within radii are summarized.
    radii : [float]
        The radii at which the mass of the mass profile is output to the summary file.
    critical_surface_mass_density : float
        The critical surface mass density of the lensing system, required to convert the mass to physical units.
    """

    for radius in radii:

        mass = mass_profile.mass_within_circle_in_mass_units(
            radius=radius, critical_surface_mass_density=critical_surface_mass_density)

        summary_file.write('Mass within {:.2f}" = {:.4e} solMass\n'.format(radius, mass))

def summarize_nfw_mass_profile(summary_file, nfw, critical_surface_mass_density_arcsec,
                               cosmic_average_mass_density_arcsec):

    rho_at_scale_radius = \
        nfw.rho_at_scale_radius(critical_surface_mass_density_arcsec=critical_surface_mass_density_arcsec)

    delta_concentration = \
        nfw.delta_concentration(critical_surface_mass_density_arcsec=critical_surface_mass_density_arcsec,
                                cosmic_average_mass_density_arcsec=cosmic_average_mass_density_arcsec)

    concentration = nfw.concentration(critical_surface_mass_density_arcsec=critical_surface_mass_density_arcsec,
                                      cosmic_average_mass_density_arcsec=cosmic_average_mass_density_arcsec)

    radius_at_200 = nfw.radius_at_200(critical_surface_mass_density_arcsec=critical_surface_mass_density_arcsec,
                                      cosmic_average_mass_density_arcsec=cosmic_average_mass_density_arcsec)

    mass_at_200 = nfw.mass_at_200(critical_surface_mass_density_arcsec=critical_surface_mass_density_arcsec,
                                  cosmic_average_mass_density_arcsec=cosmic_average_mass_density_arcsec)

    summary_file.write('Rho at scale radius = {:.2f}\n'.format(rho_at_scale_radius))
    summary_file.write('Delta concentration = {:.2f}\n'.format(delta_concentration))
    summary_file.write('Concentration = {:.2f}\n'.format(concentration))
    summary_file.write('Radius at 200x cosmic average density = {:.2f}"\n'.format(radius_at_200))
    summary_file.write('Mass at 200x cosmic average density = {:.2f} solMass\n'.format(mass_at_200))

def summarize_truncated_nfw_mass_profile(summary_file, truncated_nfw, critical_surface_mass_density_arcsec,
                                         cosmic_average_mass_density_arcsec):

    summarize_nfw_mass_profile(summary_file=summary_file, nfw=truncated_nfw,
                               critical_surface_mass_density_arcsec=critical_surface_mass_density_arcsec,
                               cosmic_average_mass_density_arcsec=cosmic_average_mass_density_arcsec)

    mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(
        critical_surface_mass_density_arcsec=critical_surface_mass_density_arcsec,
        cosmic_average_mass_density_arcsec=cosmic_average_mass_density_arcsec)

    summary_file.write('Mass at truncation radius = {:.2f} solMass\n'.format(mass_at_truncation_radius))

def summarize_truncated_nfw_challenge_mass_profile(summary_file, truncated_nfw_challenge):

    summarize_truncated_nfw_mass_profile(summary_file=summary_file, truncated_nfw=truncated_nfw_challenge,
                                         critical_surface_mass_density_arcsec=1940654909.4133248,
                                         cosmic_average_mass_density_arcsec=262.30319684750657)