import sys
import os
import re
import getdist
import numpy as np
from astropy import cosmology

sys.path.append("../autolens/")

from profiles import light_profiles, mass_profiles
import galaxy

slacs_mass = '{0:e}'.format(10**11.25)

sis = mass_profiles.SphericalIsothermalMassProfile(einstein_radius=1.03595)

lens_galaxy = galaxy.Galaxy(redshift=0.2803, mass_profiles=[sis])

source_galaxy = galaxy.Galaxy(redshift=0.9818)

galaxy.LensingPlanes(galaxies=[lens_galaxy, source_galaxy], cosmological_model=cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7))

print(4.4*lens_galaxy.arcsec_per_kpc)

print('{0:e}'.format(lens_galaxy.mass_within_circles(radius=1.03595)))
print(slacs_mass)

print(lens_galaxy.mass_within_circles(radius=1.03595)/float(slacs_mass))