from astropy import constants
import math

def arcsec_per_kpc_proper_from_redshift_and_cosmology(redshift, cosmology):
    return cosmology.arcsec_per_kpc_proper(z=redshift)

def kpc_per_arcsec_proper_from_redshift_and_cosmology(redshift, cosmology):
    return 1.0 / cosmology.arcsec_per_kpc_proper(z=redshift)

def angular_diameter_distance_to_earth_from_redshift_and_cosmology(redshift, cosmology, units_distance='arcsec'):

    angular_diameter_distance_kpc = cosmology.angular_diameter_distance(z=redshift).to('kpc')

    if units_distance is not 'arcsec':
        return angular_diameter_distance_kpc.to(units_distance)
    else:
        arcsec_per_kpc_proper = arcsec_per_kpc_proper_from_redshift_and_cosmology(redshift=redshift, cosmology=cosmology)
        return arcsec_per_kpc_proper * angular_diameter_distance_kpc

def angular_diameter_distance_between_redshifts_from_redshifts_and_cosmlology(redshift_0, redshift_1, cosmology,
                                                                              units_distance='arcsec'):

    angular_diameter_distance_between_redshifts_kpc = \
        cosmology.angular_diameter_distance_z1z2(redshift_0, redshift_1).to('kpc')

    if units_distance is not 'arcsec':
        return angular_diameter_distance_between_redshifts_kpc.to(units_distance)

def cosmic_average_mass_density_from_redshift_and_cosmology(redshift, cosmology, units_mass='solMass',
                                                            units_distance='arcsec'):

    arcsec_per_kpc_proper = arcsec_per_kpc_proper_from_redshift_and_cosmology(redshift=redshift, cosmology=cosmology)

    cosmic_average_mass_density_kpc = cosmology.critical_density(z=redshift).to(units_mass + ' / kpc^3')

    if units_distance is not 'arcsec':
        return cosmic_average_mass_density_kpc.to(units_mass + ' / ' + units_distance + '^3')
    else:
        return cosmic_average_mass_density_kpc / arcsec_per_kpc_proper ** 3.0

def critical_surface_mass_density_between_redshifts_from_redshifts_and_cosmology(
        redshift_0, redshift_1, cosmology, units_mass='solMass', units_distance='arcsec'):

    const = constants.c.to('kpc / s') ** 2.0 / (4 * math.pi * constants.G.to( 'kpc3 / (' + units_mass + ' s2)'))

    angular_diameter_distance_of_redshift_0_to_earth_kpc = \
        angular_diameter_distance_to_earth_from_redshift_and_cosmology(redshift=redshift_0, cosmology=cosmology,
                                                                       units_distance='kpc')

    angular_diameter_distance_of_redshift_1_to_earth_kpc = \
        angular_diameter_distance_to_earth_from_redshift_and_cosmology(redshift=redshift_1, cosmology=cosmology,
                                                                       units_distance='kpc')

    angular_diameter_distance_between_redshifts_kpc = \
        angular_diameter_distance_between_redshifts_from_redshifts_and_cosmlology(redshift_0=redshift_0,
                                                                                  redshift_1=redshift_1,
                                                                                  cosmology=cosmology,
                                                                                  units_distance='kpc')

    critical_surface_mass_density_kpc = ((const * angular_diameter_distance_of_redshift_1_to_earth_kpc /
                                     (angular_diameter_distance_between_redshifts_kpc *
                                      angular_diameter_distance_of_redshift_0_to_earth_kpc)))

    if units_distance is not 'arcsec':
        return critical_surface_mass_density_kpc.to(units_mass + ' / ' + units_distance + '2')
    elif units_distance is 'arcsec':
        kpc_per_arcsec_proper = kpc_per_arcsec_proper_from_redshift_and_cosmology(redshift=redshift_0,
                                                                                  cosmology=cosmology)
        return ((kpc_per_arcsec_proper**2.0) * critical_surface_mass_density_kpc)

def scaling_factor_between_redshifts_from_redshifts_and_cosmology(redshift_0, redshift_1, redshift_final, cosmology):

    angular_diameter_distance_between_redshifts_0_and_1 = \
        cosmology.angular_diameter_distance_z1z2(z1=redshift_0, z2=redshift_1).to('kpc').value

    angular_diameter_distance_to_redshift_final = \
        cosmology.angular_diameter_distance(z=redshift_final).to('kpc').value

    angular_diameter_distance_of_redshift_1_to_earth = \
        cosmology.angular_diameter_distance(z=redshift_1).to('kpc').value

    angular_diameter_distance_between_redshift_1_and_final = \
        cosmology.angular_diameter_distance_z1z2(z1=redshift_0, z2=redshift_final).to('kpc').value

    return (angular_diameter_distance_between_redshifts_0_and_1 * angular_diameter_distance_to_redshift_final) / \
           (angular_diameter_distance_of_redshift_1_to_earth * angular_diameter_distance_between_redshift_1_and_final)