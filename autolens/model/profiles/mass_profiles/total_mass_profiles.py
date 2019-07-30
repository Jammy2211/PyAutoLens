import inspect
from pyquad import quad_grid

import numpy as np
from astropy import cosmology as cosmo

import autofit as af
from autolens import dimensions as dim
from autolens.data.array import grids
from autolens.model.profiles import geometry_profiles

from autolens.model.profiles import mass_profiles as mp

from autolens.data.array.grids import reshape_returned_array, reshape_returned_grid


class PointMass(geometry_profiles.SphericalProfile, mp.MassProfile):
    @af.map_types
    def __init__(
        self, centre: dim.Position = (0.0, 0.0), einstein_radius: dim.Length = 1.0
    ):
        """
        Represents a point-mass.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius of the point-mass.
        """
        super(PointMass, self).__init__(centre=centre)
        self.einstein_radius = einstein_radius

    @reshape_returned_grid
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid, return_in_2d=True, return_binned=True):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return self.grid_to_grid_cartesian(
            grid=grid, radius=self.einstein_radius / grid_radii
        )

    # @property
    # def mass(self):
    #     return (206265 * self.einstein_radius * (constants.c**2.0) / (4.0 * constants.G)) / 1.988e30


class EllipticalCoredPowerLaw(mp.EllipticalMassProfile, mp.MassProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        einstein_radius: dim.Length = 1.0,
        slope: float = 2.0,
        core_radius: dim.Length = 0.01,
    ):
        """
        Represents a cored elliptical power-law density distribution

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            The elliptical mass profile's minor-to-major axis ratio (b/a).
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis.
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(EllipticalCoredPowerLaw, self).__init__(
            centre=centre, axis_ratio=axis_ratio, phi=phi
        )
        self.einstein_radius = einstein_radius
        self.slope = slope
        self.core_radius = core_radius

    @property
    def einstein_radius_rescaled(self):
        """Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles
        parameters"""
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius ** (
            self.slope - 1
        )

    @reshape_returned_array
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def convergence_from_grid(self, grid, return_in_2d=True, return_binned=True):
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

        The *reshape_returned_array* decorator reshapes the NumPy array the convergence is outputted on. See \
        *grids.reshape_returned_array* for a description of the output.

        Parameters
        ----------
        grid : grids.Grid
            The grid of (y,x) arc-second coordinates the surface density is computed on.
        return_in_2d : bool
            If *True*, the returned array is mapped to its unmasked 2D shape, if *False* it is the masked 1D shape.
        return_binned : bool
            If *True*, the returned array which is computed on a sub-grid is binned up to the regular grid dimensions \
            by taking the mean of all sub-gridded values. If *False*, the array is returned on the dimensions of the \
            sub-grid.
        """

        surface_density_grid = np.zeros(grid.shape[0])

        grid_eta = self.grid_to_elliptical_radii(grid)

        for i in range(grid.shape[0]):
            surface_density_grid[i] = self.convergence_func(grid_eta[i])

        return surface_density_grid

    @reshape_returned_array
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def potential_from_grid(self, grid, return_in_2d=True, return_binned=True):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

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

        potential_grid = quad_grid(
            self.potential_func,
            0.0,
            1.0,
            grid,
            args=(self.axis_ratio, self.slope, self.core_radius),
        )[0]

        return self.einstein_radius_rescaled * self.axis_ratio * potential_grid

    @reshape_returned_grid
    @grids.grid_interpolate
    @geometry_profiles.cache
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

        def calculate_deflection_component(npow, index):
            einstein_radius_rescaled = self.einstein_radius_rescaled

            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= quad_grid(
                self.deflection_func,
                0.0,
                1.0,
                grid,
                args=(
                    npow,
                    self.axis_ratio,
                    einstein_radius_rescaled,
                    self.slope,
                    self.core_radius,
                ),
            )[0]

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    def convergence_func(self, radius):
        return self.einstein_radius_rescaled * (
            self.core_radius ** 2 + radius ** 2
        ) ** (-(self.slope - 1) / 2.0)

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (
            (eta / u)
            * ((3.0 - slope) * eta) ** -1.0
            * (
                (core_radius ** 2.0 + eta ** 2.0) ** ((3.0 - slope) / 2.0)
                - core_radius ** (3 - slope)
            )
            / ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)
        )

    @staticmethod
    def deflection_func(
        u, y, x, npow, axis_ratio, einstein_radius_rescaled, slope, core_radius
    ):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (
            einstein_radius_rescaled
            * (core_radius ** 2 + eta_u ** 2) ** (-(slope - 1) / 2.0)
            / ((1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))
        )

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)

    @dim.convert_units_to_input_units
    def summarize_in_units(
        self,
        radii,
        prefix="",
        whitespace=80,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        summary = super().summarize_in_units(
            radii=radii,
            prefix=prefix,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            cosmology=cosmology,
            whitespace=whitespace,
            kwargs=kwargs,
        )

        return summary

    @property
    def unit_mass(self):
        return "angular"


class SphericalCoredPowerLaw(EllipticalCoredPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
        slope: float = 2.0,
        core_radius: dim.Length = 0.01,
    ):
        """
        Represents a cored spherical power-law density distribution

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(SphericalCoredPowerLaw, self).__init__(
            centre=centre,
            axis_ratio=1.0,
            phi=0.0,
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=core_radius,
        )

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
        eta = self.grid_to_grid_radii(grid=grid)
        deflection = np.multiply(
            2.0 * self.einstein_radius_rescaled,
            np.divide(
                np.add(
                    np.power(
                        np.add(self.core_radius ** 2, np.square(eta)),
                        (3.0 - self.slope) / 2.0,
                    ),
                    -self.core_radius ** (3 - self.slope),
                ),
                np.multiply((3.0 - self.slope), eta),
            ),
        )
        return self.grid_to_grid_cartesian(grid=grid, radius=deflection)


class EllipticalPowerLaw(EllipticalCoredPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        einstein_radius: dim.Length = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents an elliptical power-law density distribution.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            The elliptical mass profile's minor-to-major axis ratio (b/a).
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis.
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super(EllipticalPowerLaw, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            phi=phi,
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=dim.Length(0.0),
        )

    def convergence_func(self, radius):
        if radius > 0.0:
            return self.einstein_radius_rescaled * radius ** (-(self.slope - 1))
        else:
            return np.inf

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (
            (eta_u / u)
            * ((3.0 - slope) * eta_u) ** -1.0
            * eta_u ** (3.0 - slope)
            / ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)
        )

    @staticmethod
    def deflection_func(
        u, y, x, npow, axis_ratio, einstein_radius_rescaled, slope, core_radius
    ):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        return (
            einstein_radius_rescaled
            * eta_u ** (-(slope - 1))
            / ((1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))
        )


class SphericalPowerLaw(EllipticalPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents a spherical power-law density distribution.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super(SphericalPowerLaw, self).__init__(
            centre=centre,
            axis_ratio=1.0,
            phi=0.0,
            einstein_radius=einstein_radius,
            slope=slope,
        )

    @reshape_returned_grid
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid, return_in_2d=True, return_binned=True):
        eta = self.grid_to_grid_radii(grid)
        deflection_r = (
            2.0
            * self.einstein_radius_rescaled
            * np.divide(
                np.power(eta, (3.0 - self.slope)), np.multiply((3.0 - self.slope), eta)
            )
        )
        return self.grid_to_grid_cartesian(grid, deflection_r)


class EllipticalCoredIsothermal(EllipticalCoredPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        einstein_radius: dim.Length = 1.0,
        core_radius: dim.Length = 0.01,
    ):
        """
        Represents a cored elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            The elliptical mass profile's minor-to-major axis ratio (b/a).
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis.
        einstein_radius : float
            The arc-second Einstein radius.
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(EllipticalCoredIsothermal, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            phi=phi,
            einstein_radius=einstein_radius,
            slope=2.0,
            core_radius=core_radius,
        )


class SphericalCoredIsothermal(SphericalCoredPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        einstein_radius: dim.Length = 1.0,
        core_radius: dim.Length = 0.01,
    ):
        """
        Represents a cored spherical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(SphericalCoredIsothermal, self).__init__(
            centre=centre,
            einstein_radius=einstein_radius,
            slope=2.0,
            core_radius=core_radius,
        )


class EllipticalIsothermal(EllipticalPowerLaw):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        einstein_radius: dim.Length = 1.0,
    ):
        """
        Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            The elliptical mass profile's minor-to-major axis ratio (b/a).
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis.
        einstein_radius : float
            The arc-second Einstein radius.
        """
        super(EllipticalIsothermal, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            phi=phi,
            einstein_radius=einstein_radius,
            slope=2.0,
        )

    # @classmethod
    # def from_mass_in_solar_masses(cls, redshift_lens=0.5, redshift_source=1.0, centre: units.Position = (0.0, 0.0), axis_ratio_=0.9,
    #                               phi: float = 0.0, mass=10e10):
    #
    #     return self.constant_kpc * self.angular_diameter_distance_of_plane_to_earth(j) / \
    #            (self.angular_diameter_distance_between_planes(i, j) *
    #             self.angular_diameter_distance_of_plane_to_earth(i))

    # critical_surface_density =

    @reshape_returned_grid
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid, return_in_2d=True, return_binned=True):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore, \
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

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
        factor = (
            2.0
            * self.einstein_radius_rescaled
            * self.axis_ratio
            / np.sqrt(1 - self.axis_ratio ** 2)
        )

        psi = np.sqrt(
            np.add(
                np.multiply(self.axis_ratio ** 2, np.square(grid[:, 1])),
                np.square(grid[:, 0]),
            )
        )

        deflection_y = np.arctanh(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 0]), psi)
        )
        deflection_x = np.arctan(
            np.divide(np.multiply(np.sqrt(1 - self.axis_ratio ** 2), grid[:, 1]), psi)
        )
        return self.rotate_grid_from_profile(
            np.multiply(factor, np.vstack((deflection_y, deflection_x)).T)
        )


class SphericalIsothermal(EllipticalIsothermal):
    @af.map_types
    def __init__(
        self, centre: dim.Position = (0.0, 0.0), einstein_radius: dim.Length = 1.0
    ):
        """
        Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius : float
            The arc-second Einstein radius.
        """
        super(SphericalIsothermal, self).__init__(
            centre=centre, axis_ratio=1.0, phi=0.0, einstein_radius=einstein_radius
        )

    @reshape_returned_array
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def potential_from_grid(self, grid, return_in_2d=True, return_binned=True):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

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
        eta = self.grid_to_elliptical_radii(grid)
        return 2.0 * self.einstein_radius_rescaled * eta

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
        return self.grid_to_grid_cartesian(
            grid=grid,
            radius=np.full(grid.shape[0], 2.0 * self.einstein_radius_rescaled),
        )


class EllipticalIsothermalKormann(mp.EllipticalMassProfile, mp.MassProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        einstein_radius: dim.Length = 1.0,
    ):
        """
        Represents a cored elliptical power-law density distribution

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            The elliptical mass profile's minor-to-major axis ratio (b/a).
        phi : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive x-axis.
        einstein_radius : float
            The arc-second Einstein radius.
        slope : float
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius : float
            The arc-second radius of the inner core.
        """
        super(EllipticalIsothermalKormann, self).__init__(
            centre=centre, axis_ratio=axis_ratio, phi=phi
        )
        self.einstein_radius = einstein_radius

    @reshape_returned_array
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def convergence_from_grid(self, grid, return_in_2d=True, return_binned=True):
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

        The *reshape_returned_array* decorator reshapes the NumPy array the convergence is outputted on. See \
        *grids.reshape_returned_array* for a description of the output.

        Parameters
        ----------
        grid : grids.Grid
            The grid of (y,x) arc-second coordinates the surface density is computed on.
        return_in_2d : bool
            If *True*, the returned array is mapped to its unmasked 2D shape, if *False* it is the masked 1D shape.
        return_binned : bool
            If *True*, the returned array which is computed on a sub-grid is binned up to the regular grid dimensions \
            by taking the mean of all sub-gridded values. If *False*, the array is returned on the dimensions of the \
            sub-grid.
        """

        surface_density_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            surface_density_grid[i] = self.convergence_func(y=grid[i,0], x=grid[i,1])

        return surface_density_grid

    @reshape_returned_array
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def potential_from_grid(self, grid, return_in_2d=True, return_binned=True):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

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
        f_prime = np.sqrt(1 - self.axis_ratio ** 2)
        sin_phi = grid[:,1] / np.sqrt(self.axis_ratio ** 2 * grid[:,0] ** 2 + grid[:,1] ** 2)
        cos_phi = grid[:,0] / np.sqrt(self.axis_ratio ** 2 * grid[:,0] ** 2 + grid[:,1] ** 2)
        return (np.sqrt(self.axis_ratio) / f_prime) * (grid[:,1] * np.arcsin(f_prime * sin_phi) + grid[:,0] * np.arcsinh((f_prime / self.axis_ratio) * cos_phi))

    @reshape_returned_grid
    @grids.grid_interpolate
    @geometry_profiles.cache
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

        deflection_y = np.arcsin((np.sqrt(1 - self.axis_ratio ** 2) * grid[:,0]) / (np.sqrt((grid[:,1] ** 2) * (self.axis_ratio ** 2) + grid[:,0] ** 2)))
        deflection_x = np.arcsinh((np.sqrt(1 - self.axis_ratio ** 2) * grid[:,1]) / (self.axis_ratio * np.sqrt((self.axis_ratio ** 2) * (grid[:,1] ** 2) + grid[:,0] ** 2)))

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    def convergence_func(self, y, x):
        return self.einstein_radius * np.sqrt(self.axis_ratio)/(2*np.sqrt(x**2*self.axis_ratio**2+y**2))

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)

    @dim.convert_units_to_input_units
    def summarize_in_units(
        self,
        radii,
        prefix="",
        whitespace=80,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        summary = super().summarize_in_units(
            radii=radii,
            prefix=prefix,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            cosmology=cosmology,
            whitespace=whitespace,
            kwargs=kwargs,
        )

        return summary

    @property
    def unit_mass(self):
        return "angular"