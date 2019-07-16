import numpy as np
from astropy import cosmology as cosmo

import autofit as af
from autolens import dimensions as dim
from autolens.data.array import grids
from autolens.model.profiles import geometry_profiles
from autolens.model.profiles import mass_profiles as mp

from autolens.data.array.grids import reshape_returned_array, reshape_returned_grid


class MassSheet(geometry_profiles.SphericalProfile, mp.MassProfile):
    @af.map_types
    def __init__(self, centre: dim.Position = (0.0, 0.0), kappa: float = 0.0):
        """
        Represents a mass-sheet

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        kappa : float
            The magnitude of the convergence of the mass-sheet.
        """
        super(MassSheet, self).__init__(centre=centre)
        self.kappa = kappa

    @reshape_returned_array
    def convergence_from_grid(self, grid, return_in_2d=True, return_binned=True):
        return np.full(shape=grid.shape[0], fill_value=self.kappa)

    @reshape_returned_array
    def potential_from_grid(self, grid, return_in_2d=True, return_binned=True):
        return np.zeros((grid.shape[0],))

    @reshape_returned_grid
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid, return_in_2d=True, return_binned=True):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return self.grid_to_grid_cartesian(grid=grid, radius=self.kappa * grid_radii)


# noinspection PyAbstractClass
class ExternalShear(geometry_profiles.EllipticalProfile, mp.MassProfile):
    @af.map_types
    def __init__(self, magnitude: float = 0.2, phi: float = 0.0):
        """
        An external shear term, to model the line-of-sight contribution of other galaxies / satellites.

        The shear angle phi is defined in the direction of stretching of the image. Therefore, if an object located \
        outside the lens is responsible for the shear, it will be offset 90 degrees from the value of phi.

        Parameters
        ----------
        magnitude : float
            The overall magnitude of the shear (gamma).
        phi : float
            The rotation axis of the shear.
        """

        super(ExternalShear, self).__init__(centre=(0.0, 0.0), phi=phi, axis_ratio=1.0)
        self.magnitude = magnitude

    def einstein_radius_in_units(
        self,
        unit_mass="solMass",
        redshift_profile=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        return 0.0

    def einstein_mass_in_units(
        self,
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        return 0.0

    @reshape_returned_array
    def convergence_from_grid(self, grid, return_in_2d=True, return_binned=True):
        return np.zeros((grid.shape[0],))

    @reshape_returned_array
    def potential_from_grid(self, grid, return_in_2d=True, return_binned=True):
        return np.zeros((grid.shape[0],))

    @reshape_returned_grid
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid, return_in_2d=True, return_binned=True):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        return_in_2d : bool
            If *True*, the returned array is mapped to its unmasked 2D shape, if *False* it is the masked 1D shape.
        return_binned : bool
            If *True*, the returned array which is computed on a sub-grid is binned up to the regular grid dimensions \
            by taking the mean of all sub-gridded values. If *False*, the array is returned on the dimensions of the \
            sub-grid.
        """
        deflection_y = -np.multiply(self.magnitude, grid[:, 0])
        deflection_x = np.multiply(self.magnitude, grid[:, 1])
        return self.rotate_grid_from_profile(np.vstack((deflection_y, deflection_x)).T)
