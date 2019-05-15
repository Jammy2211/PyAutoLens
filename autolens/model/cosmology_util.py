from astropy import constants
from autolens import exc, dimensions as dim
import math

def arcsec_per_kpc_from_redshift_and_cosmology(redshift, cosmology):
    return cosmology.arcsec_per_kpc_proper(z=redshift).value

def kpc_per_arcsec_from_redshift_and_cosmology(redshift, cosmology):
    return 1.0 / cosmology.arcsec_per_kpc_proper(z=redshift).value

def angular_diameter_distance_to_earth_from_redshift_and_cosmology(redshift, cosmology, unit_length='arcsec'):

    angular_diameter_distance_kpc = cosmology.angular_diameter_distance(z=redshift).to('kpc')

    if unit_length is not 'arcsec':
        angular_diameter_distance = angular_diameter_distance_kpc.to(unit_length)
    else:
        arcsec_per_kpc = arcsec_per_kpc_from_redshift_and_cosmology(redshift=redshift, cosmology=cosmology)
        angular_diameter_distance = arcsec_per_kpc * angular_diameter_distance_kpc

    return dim.Length(angular_diameter_distance.value, unit_length)

def angular_diameter_distance_between_redshifts_from_redshifts_and_cosmlology(redshift_0, redshift_1, cosmology,
                                                                              unit_length='kpc'):

    if unit_length is 'arcsec':
        raise exc.UnitsException('The angular diameter distance between redshifts cannot have units of length')

    angular_diameter_distance_between_redshifts_kpc = \
        cosmology.angular_diameter_distance_z1z2(redshift_0, redshift_1).to('kpc')

    if unit_length is not 'arcsec':
        angular_diameter_distance = angular_diameter_distance_between_redshifts_kpc.to(unit_length)

    return dim.Length(angular_diameter_distance.value, unit_length)

def cosmic_average_density_from_redshift_and_cosmology(redshift, cosmology,unit_length='arcsec',
                                                            unit_mass='solMass'):

    cosmic_average_density_kpc = cosmology.critical_density(z=redshift).to(unit_mass + ' / kpc^3')

    cosmic_average_density_kpc = dim.MassOverLength3(value=cosmic_average_density_kpc.value,
                                                          unit_length='kpc', unit_mass=unit_mass)

    if unit_length is not 'arcsec':
        cosmic_average_density = cosmic_average_density_kpc.convert(unit_length=unit_length,
                                                                              unit_mass=unit_mass)
    else:
        kpc_per_arcsec = kpc_per_arcsec_from_redshift_and_cosmology(redshift=redshift,
                                                                           cosmology=cosmology)
        cosmic_average_density = cosmic_average_density_kpc.convert(unit_length=unit_length,
                                                                              unit_mass=unit_mass,
                                                                              kpc_per_arcsec=kpc_per_arcsec)

    return cosmic_average_density

def critical_surface_density_between_redshifts_from_redshifts_and_cosmology(
        redshift_0, redshift_1, cosmology, unit_length='arcsec', unit_mass='solMass'):

    if unit_mass is 'angular':
        return dim.MassOverLength2(value=1.0, unit_mass=unit_mass, unit_length=unit_length)

    const = constants.c.to('kpc / s') ** 2.0 / (4 * math.pi * constants.G.to( 'kpc3 / (' + unit_mass + ' s2)'))

    angular_diameter_distance_of_redshift_0_to_earth_kpc = \
        angular_diameter_distance_to_earth_from_redshift_and_cosmology(redshift=redshift_0, cosmology=cosmology,
                                                                       unit_length='kpc')

    angular_diameter_distance_of_redshift_1_to_earth_kpc = \
        angular_diameter_distance_to_earth_from_redshift_and_cosmology(redshift=redshift_1, cosmology=cosmology,
                                                                       unit_length='kpc')

    angular_diameter_distance_between_redshifts_kpc = \
        angular_diameter_distance_between_redshifts_from_redshifts_and_cosmlology(redshift_0=redshift_0,
                                                                                  redshift_1=redshift_1,
                                                                                  cosmology=cosmology,
                                                                                  unit_length='kpc')

    critical_surface_density_kpc = ((const * angular_diameter_distance_of_redshift_1_to_earth_kpc /
                                     (angular_diameter_distance_between_redshifts_kpc *
                                      angular_diameter_distance_of_redshift_0_to_earth_kpc)))

    critical_surface_density_kpc = dim.MassOverLength2(value=critical_surface_density_kpc.value,
                                                            unit_mass=unit_mass, unit_length='kpc')

    if unit_length is not 'arcsec':
        critical_surface_density = critical_surface_density_kpc.convert(unit_length=unit_length,
                                                                                  unit_mass=unit_mass)

    elif unit_length is 'arcsec':
        kpc_per_arcsec = kpc_per_arcsec_from_redshift_and_cosmology(redshift=redshift_0,
                                                                                  cosmology=cosmology)
        critical_surface_density = critical_surface_density_kpc.convert(unit_mass=unit_mass,
                                                                                  unit_length=unit_length,
                                                                                  kpc_per_arcsec=kpc_per_arcsec)

    return critical_surface_density

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