from profile import light_profile, mass_profile


class GalaxyCollection(list):
    """A collection of galaxies ordered by redshift"""

    def append(self, galaxy):
        """
        Append a new galaxy to the collection in the correct position according to its redshift.

        Parameters
        ----------
        galaxy: Galaxy
            A galaxy
        """

        def insert(position):
            if position == len(self):
                super(GalaxyCollection, self).append(galaxy)
            elif galaxy.redshift <= self[position].redshift:
                self[:] = self[:position] + [galaxy] + self[position:]
            else:
                insert(position + 1)

        insert(0)


class Galaxy(object):
    """Represents a real galaxy. This could be a lens galaxy or source galaxy. Note that a lens galaxy must have mass
    profiles"""

    def __init__(self, redshift, light_profiles=None, mass_profiles=None):
        """
        Parameters
        ----------
        redshift: float
            The redshift of this galaxy
        light_profiles: [LightProfile]
            A list of light profiles describing the light profile of this galaxy
        mass_profiles: [MassProfile]
            A list of mass profiles describing the mass profile of this galaxy
        """
        self.redshift = redshift
        self.light_profiles = light_profile.CombinedLightProfile(
            *light_profiles) if light_profiles is not None else light_profile.CombinedLightProfile()
        self.mass_profiles = mass_profile.CombinedMassProfile(
            *mass_profiles) if mass_profiles is not None else mass_profile.CombinedMassProfile()

    def intensity_at_coordinates(self, coordinates):
        """
        Method for obtaining the summed light profiles' intensities at a given set of coordinates
        Parameters
        ----------
        coordinates : (float, float)
            The coordinates in image space
        Returns
        -------
        intensity : float
            The summed values of intensity at the given coordinates
        """
        return self.light_profiles.intensity_at_coordinates(coordinates)

    def surface_density_at_coordinates(self, coordinates):
        """
        Method for obtaining the summed mass profiles' surface densities at a given set of coordinates.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The summed values of surface density at the given coordinates.
        """

        return self.mass_profiles.surface_density_at_coordinates(coordinates)

    def potential_at_coordinates(self, coordinates):
        """
        Method for obtaining the summed mass profiles' gravitational potential at a given set of coordinates.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The summed values of gravitational potential at the given coordinates.
        """

        return self.mass_profiles.potential_at_coordinates(coordinates)

    def deflection_angles_at_coordinates(self, coordinates):
        """
        Method for obtaining the summed mass profiles' deflection angles at a given set of coordinates.

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the image

        Returns
        ----------
        The summed values of deflection angles at the given coordinates.
        """
        return self.mass_profiles.deflection_angles_at_coordinates(coordinates)

    def __repr__(self):
        return "<Galaxy redshift={}>".format(self.redshift)
