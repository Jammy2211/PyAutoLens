import math
from astropy import constants

class Value(object):

    def __init__(self, value):

        self.value = value

    def to(self, *args, **kwargs):
        return Value(value=self.value)

class MockCosmology(object):

    def __init__(self, arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0, cosmic_average_density=2.0):

        self.arcsec_per_kpc = arcsec_per_kpc
        self.kpc_per_arcsec = kpc_per_arcsec
        self.critical_surface_density = critical_surface_density
        self.cosmic_average_density = cosmic_average_density

    def arcsec_per_kpc_proper(self, z):
        return Value(value=self.arcsec_per_kpc)

    def kpc_per_arcsec_proper(self, z):
        return Value(value=self.kpc_per_arcsec)

    def angular_diameter_distance(self, z):
        return Value(value=1.0)

    def angular_diameter_distance_z1z2(self, z1, z2):
        const = constants.c.to('kpc / s') ** 2.0 / (4 * math.pi * constants.G.to('kpc3 / (solMass s2)'))
        return Value(value=self.critical_surface_density * const.value)

    def critical_density(self, z):
        return Value(value=self.cosmic_average_density)