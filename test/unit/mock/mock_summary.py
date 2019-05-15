from astropy import cosmology as cosmo
from autolens.model.profiles import mass_profiles as mp

class MockMassProfile(object):

    def __init__(self):
        pass

    @property
    def einstein_radius(self):
        return 10.0

    def mass_within_circle(self, radius, mass_units=None, critical_surface_density=None):
        return 1000.0


class MockNFW(MockMassProfile):

    def __init__(self):
        super(MockNFW, self).__init__()
        self.scale_radius = 1.0

    def rho_at_scale_radius(self, critical_surface_density_arcsec):
        return 100.0

    def delta_concentration(self, critical_surface_density_arcsec, cosmic_average_density_arcsec):
        return 200.0

    def concentration(self, critical_surface_density_arcsec, cosmic_average_density_arcsec):
        return 300.0

    def radius_at_200(self, critical_surface_density_arcsec, cosmic_average_density_arcsec):
        return 400.0

    def mass_at_200(self, critical_surface_density_arcsec, cosmic_average_density_arcsec):
        return 500.0


class MockTruncatedNFW(MockNFW, mp.SphericalTruncatedNFWChallenge):

    def __init__(self):
        super(MockTruncatedNFW, self).__init__()

    def mass_at_truncation_radius(self, critical_surface_density, cosmic_average_density):
        return 600.0


class MockGalaxy(object):

    def __init__(self, mass_profiles):

        self.mass_profiles = mass_profiles


class MockPlane(object):

    def __init__(self, galaxies):

        self.redshift = 0.5
        self.galaxies = galaxies

    @property
    def cosmic_average_density_arcsec(self):
        return 1.0


class MockTracer(object):

    def __init__(self, planes):

        self.planes = planes
        self.plane_redshifts = [0.5]

    @property
    def critical_surface_density_arcsec(self):
        return 2.0

    @property
    def cosmology(self):
        return cosmo.Planck15