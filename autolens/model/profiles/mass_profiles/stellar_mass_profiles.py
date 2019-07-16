from pyquad import quad_grid

import numpy as np

import autofit as af
from autolens import dimensions as dim
from autolens.data.array import grids
from autolens.model.profiles import geometry_profiles

from autolens.model.profiles import mass_profiles as mp

from autolens.data.array.grids import reshape_returned_array, reshape_returned_grid

# noinspection PyAbstractClass
class AbstractEllipticalSersic(mp.EllipticalMassProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
    ):
        """
        The Sersic mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens \
        model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(AbstractEllipticalSersic, self).__init__(
            centre=centre, axis_ratio=axis_ratio, phi=phi
        )
        super(mp.EllipticalMassProfile, self).__init__(
            centre=centre, axis_ratio=axis_ratio, phi=phi
        )
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @reshape_returned_array
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def convergence_from_grid(self, grid, return_in_2d=True, return_binned=True):
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

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
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

    def convergence_func(self, radius):
        return self.mass_to_light_ratio * self.intensity_at_radius(radius)

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)

    def intensity_at_radius(self, radius):
        """ Compute the intensity of the profile at a given radius.

        Parameters
        ----------
        radius : float
            The distance from the centre of the profile.
        """
        return self.intensity * np.exp(
            -self.sersic_constant
            * (((radius / self.effective_radius) ** (1.0 / self.sersic_index)) - 1)
        )

    @property
    def sersic_constant(self):
        """ A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
        total integrated light.
        """
        return (
            (2 * self.sersic_index)
            - (1.0 / 3.0)
            + (4.0 / (405.0 * self.sersic_index))
            + (46.0 / (25515.0 * self.sersic_index ** 2))
            + (131.0 / (1148175.0 * self.sersic_index ** 3))
            - (2194697.0 / (30690717750.0 * self.sersic_index ** 4))
        )

    @property
    def elliptical_effective_radius(self):
        """The effective_radius of a Sersic light profile is defined as the circular effective radius. This is the \
        radius within which a circular aperture contains half the profiles's total integrated light. For elliptical \
        systems, this won't robustly capture the light profile's elliptical shape.

        The elliptical effective radius instead describes the major-axis radius of the ellipse containing \
        half the light, and may be more appropriate for highly flattened systems like disk galaxies."""
        return self.effective_radius / np.sqrt(self.axis_ratio)

    @property
    def unit_mass(self):
        return self.mass_to_light_ratio.unit_mass


class EllipticalSersic(AbstractEllipticalSersic):
    @staticmethod
    def deflection_func(
        u,
        y,
        x,
        npow,
        axis_ratio,
        intensity,
        sersic_index,
        effective_radius,
        mass_to_light_ratio,
        sersic_constant,
    ):
        eta_u = np.sqrt(axis_ratio) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u))))
        )

        return (
            mass_to_light_ratio
            * intensity
            * np.exp(
                -sersic_constant
                * (((eta_u / effective_radius) ** (1.0 / sersic_index)) - 1)
            )
            / ((1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))
        )

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
            sersic_constant = self.sersic_constant

            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= quad_grid(
                self.deflection_func,
                0.0,
                1.0,
                grid,
                args=(
                    npow,
                    self.axis_ratio,
                    self.intensity,
                    self.sersic_index,
                    self.effective_radius,
                    self.mass_to_light_ratio,
                    sersic_constant,
                ),
            )[0]

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )


class SphericalSersic(EllipticalSersic):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
    ):
        """
        The Sersic mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens
        model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        """
        super(SphericalSersic, self).__init__(
            centre=centre,
            axis_ratio=1.0,
            phi=0.0,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllipticalExponential(EllipticalSersic):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
    ):
        """
        The EllipticalExponential mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles
        """
        super(EllipticalExponential, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            phi=phi,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SphericalExponential(EllipticalExponential):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
    ):
        """
        The Exponential mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens
        model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles.
        """
        super(SphericalExponential, self).__init__(
            centre=centre,
            axis_ratio=1.0,
            phi=0.0,
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllipticalDevVaucouleurs(EllipticalSersic):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
    ):
        """
        The EllipticalDevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        """
        super(EllipticalDevVaucouleurs, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            phi=phi,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=4.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SphericalDevVaucouleurs(EllipticalDevVaucouleurs):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
    ):
        """
        The DevVaucouleurs mass profile, the mass profiles of the light profiles that are used to fit and subtract the
        lens model_galaxy's light.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profiles.
        """
        super(SphericalDevVaucouleurs, self).__init__(
            centre=centre,
            axis_ratio=1.0,
            phi=0.0,
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class EllipticalSersicRadialGradient(AbstractEllipticalSersic):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient : float
            The mass-to-light radial gradient.
        """
        super(EllipticalSersicRadialGradient, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            phi=phi,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )
        self.mass_to_light_gradient = mass_to_light_gradient

    @reshape_returned_array
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def convergence_from_grid(self, grid, return_in_2d=True, return_binned=True):
        """ Calculate the projected convergence at a given set of arc-second gridded coordinates.

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
        return self.convergence_func(self.grid_to_eccentric_radii(grid))

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
            sersic_constant = self.sersic_constant

            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= quad_grid(
                self.deflection_func,
                0.0,
                1.0,
                grid,
                args=(
                    npow,
                    self.axis_ratio,
                    self.intensity,
                    self.sersic_index,
                    self.effective_radius,
                    self.mass_to_light_ratio,
                    self.mass_to_light_gradient,
                    sersic_constant,
                ),
            )[0]
            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    def convergence_func(self, radius):
        return (
            self.mass_to_light_ratio
            * (
                ((self.axis_ratio * radius) / self.effective_radius)
                ** -self.mass_to_light_gradient
            )
            * self.intensity_at_radius(radius)
        )

    @staticmethod
    def deflection_func(
        u,
        y,
        x,
        npow,
        axis_ratio,
        intensity,
        sersic_index,
        effective_radius,
        mass_to_light_ratio,
        mass_to_light_gradient,
        sersic_constant,
    ):
        eta_u = np.sqrt(axis_ratio) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u))))
        )

        return (
            mass_to_light_ratio
            * (((axis_ratio * eta_u) / effective_radius) ** -mass_to_light_gradient)
            * intensity
            * np.exp(
                -sersic_constant
                * (((eta_u / effective_radius) ** (1.0 / sersic_index)) - 1)
            )
            / ((1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))
        )


class SphericalSersicRadialGradient(EllipticalSersicRadialGradient):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        effective_radius: dim.Length = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: dim.MassOverLuminosity = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius : float
            The circular radius containing half the light of this profile.
        sersic_index : float
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        mass_to_light_ratio : float
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient : float
            The mass-to-light radial gradient.
        """
        super(SphericalSersicRadialGradient, self).__init__(
            centre=centre,
            axis_ratio=1.0,
            phi=0.0,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )
