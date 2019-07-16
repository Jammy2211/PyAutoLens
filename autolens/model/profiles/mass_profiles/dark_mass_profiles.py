from pyquad import quad_grid

import numpy as np
from scipy import special
from scipy.integrate import quad
from scipy.optimize import fsolve
from astropy import cosmology as cosmo

import inspect
from numba import cfunc
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable
from scipy import special

import autofit as af
from autolens import dimensions as dim
from autolens import decorator_util
from autolens.data.array import grids
from autolens.model.profiles import geometry_profiles
from autolens.model.profiles import mass_profiles as mp

from autolens.data.array.grids import reshape_returned_array, reshape_returned_grid


def jit_integrand(integrand_function):

    jitted_function = decorator_util.jit(nopython=True, cache=True)(integrand_function)
    no_args = len(inspect.getfullargspec(integrand_function).args)

    wrapped = None

    if no_args == 4:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3])

    elif no_args == 5:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4])

    elif no_args == 6:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5])

    elif no_args == 7:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6])

    elif no_args == 8:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(
                xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7]
            )

    elif no_args == 9:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(
                xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8]
            )

    elif no_args == 10:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(
                xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8], xx[9]
            )

    elif no_args == 11:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(
                xx[0],
                xx[1],
                xx[2],
                xx[3],
                xx[4],
                xx[5],
                xx[6],
                xx[7],
                xx[8],
                xx[9],
                xx[10],
            )

    cf = cfunc(float64(intc, CPointer(float64)))

    return LowLevelCallable(cf(wrapped).ctypes)


# noinspection PyAbstractClass
class AbstractEllipticalGeneralizedNFW(mp.EllipticalMassProfile, mp.MassProfile):
    epsrel = 1.49e-5

    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        kappa_s: float = 0.05,
        inner_slope: float = 1.0,
        scale_radius: dim.Length = 1.0,
    ):
        """
        The elliptical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        inner_slope : float
            The inner slope of the dark matter halo
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(AbstractEllipticalGeneralizedNFW, self).__init__(
            centre=centre, axis_ratio=axis_ratio, phi=phi
        )
        super(mp.MassProfile, self).__init__()
        self.kappa_s = kappa_s
        self.scale_radius = scale_radius
        self.inner_slope = inner_slope

    def tabulate_integral(self, grid, tabulate_bins):
        """Tabulate an integral over the surface density of deflection potential of a mass profile. This is used in \
        the GeneralizedNFW profile classes to speed up the integration procedure.

        Parameters
        -----------
        grid : grids.Grid
            The grid of (y,x) arc-second coordinates the potential / deflection_stacks are computed on.
        tabulate_bins : int
            The number of bins to tabulate the inner integral of this profile.
        """
        eta_min = 1.0e-4
        eta_max = 1.05 * np.max(self.grid_to_elliptical_radii(grid))

        minimum_log_eta = np.log10(eta_min)
        maximum_log_eta = np.log10(eta_max)
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)

        return eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size

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

        surface_density_grid = np.zeros(shape=grid.shape[0])

        grid_eta = self.grid_to_elliptical_radii(grid)

        for i in range(grid.shape[0]):
            surface_density_grid[i] = self.convergence_func(grid_eta[i])

        return surface_density_grid

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio) / 2.0)

    @staticmethod
    def coord_func_f(grid_radius):
        f = np.where(
            np.real(grid_radius) > 1.0,
            (1.0 / np.sqrt(np.square(grid_radius) - 1.0))
            * np.arccos(np.divide(1.0, grid_radius)),
            (1.0 / np.sqrt(1.0 - np.square(grid_radius)))
            * np.arccosh(np.divide(1.0, grid_radius)),
        )
        f[np.isnan(f)] = 1.0
        return f

    def coord_func_g(self, grid_radius):
        f_r = self.coord_func_f(grid_radius=grid_radius)

        g = np.where(
            np.real(grid_radius) > 1.0,
            (1.0 - f_r) / (np.square(grid_radius) - 1.0),
            (f_r - 1.0) / (1.0 - np.square(grid_radius)),
        )
        g[np.isnan(g)] = 1.0 / 3.0
        return g

    def coord_func_h(self, grid_radius):
        return np.log(grid_radius / 2.0) + self.coord_func_f(grid_radius=grid_radius)

    @dim.convert_units_to_input_units
    def rho_at_scale_radius_for_units(
        self,
        redshift_profile,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        """The Cosmic average density is defined at the redshift of the profile."""

        kpc_per_arcsec = (
            kwargs["kpc_per_arcsec"] if "kpc_per_arcsec" in kwargs else None
        )
        critical_surface_density = (
            kwargs["critical_surface_density"]
            if "critical_surface_density" in kwargs
            else None
        )

        rho_at_scale_radius = (
            self.kappa_s * critical_surface_density / self.scale_radius
        )

        rho_at_scale_radius = dim.MassOverLength3(
            value=rho_at_scale_radius, unit_length=unit_length, unit_mass=unit_mass
        )

        return rho_at_scale_radius.convert(
            unit_length=unit_length,
            unit_mass=unit_mass,
            kpc_per_arcsec=kpc_per_arcsec,
            critical_surface_density=critical_surface_density,
        )

    @dim.convert_units_to_input_units
    def delta_concentration_for_units(
        self,
        redshift_profile,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        cosmic_average_density = (
            kwargs["cosmic_average_density"]
            if "cosmic_average_density" in kwargs
            else None
        )

        rho_scale_radius = self.rho_at_scale_radius_for_units(
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        return rho_scale_radius / cosmic_average_density

    @dim.convert_units_to_input_units
    def concentration_for_units(
        self,
        redshift_profile,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        delta_concentration = self.delta_concentration_for_units(
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            unit_length=unit_length,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            unit_mass=unit_mass,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        return fsolve(
            func=self.concentration_func, x0=10.0, args=(delta_concentration,)
        )[0]

    @staticmethod
    def concentration_func(concentration, delta_concentration):
        return (
            200.0
            / 3.0
            * (
                concentration
                * concentration
                * concentration
                / (np.log(1 + concentration) - concentration / (1 + concentration))
            )
            - delta_concentration
        )

    @dim.convert_units_to_input_units
    def radius_at_200_for_units(
        self,
        redshift_profile,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        kpc_per_arcsec = (
            kwargs["kpc_per_arcsec"] if "kpc_per_arcsec" in kwargs else None
        )

        concentration = self.concentration_for_units(
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        radius_at_200 = dim.Length(
            value=concentration * self.scale_radius, unit_length=unit_length
        )

        return radius_at_200.convert(
            unit_length=unit_length, kpc_per_arcsec=kpc_per_arcsec
        )

    @dim.convert_units_to_input_units
    def mass_at_200_for_units(
        self,
        redshift_profile,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        cosmic_average_density = (
            kwargs["cosmic_average_density"]
            if "cosmic_average_density" in kwargs
            else None
        )
        critical_surface_density = (
            kwargs["critical_surface_density"]
            if "critical_surface_density" in kwargs
            else None
        )

        radius_at_200 = self.radius_at_200_for_units(
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            unit_length=unit_length,
            unit_mass=unit_mass,
            cosmology=cosmology,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            kwargs=kwargs,
        )

        mass_at_200 = dim.Mass(
            200.0
            * ((4.0 / 3.0) * np.pi)
            * cosmic_average_density
            * (radius_at_200 ** 3.0),
            unit_mass=unit_mass,
        )

        return mass_at_200.convert(
            unit_mass=unit_mass, critical_surface_density=critical_surface_density
        )

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
        redshift_of_cosmic_average_density="profile",
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

        rho_at_scale_radius = self.rho_at_scale_radius_for_units(
            radii=radii,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        summary += [
            af.text_util.label_value_and_unit_string(
                label=prefix + "rho_at_scale_radius",
                value=rho_at_scale_radius,
                unit=unit_mass + "/" + unit_length + "3",
                whitespace=whitespace,
            )
        ]

        delta_concentration = self.delta_concentration_for_units(
            radii=radii,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        summary += [
            af.text_util.label_and_value_string(
                label=prefix + "delta_concentration",
                value=delta_concentration,
                whitespace=whitespace,
            )
        ]

        concentration = self.concentration_for_units(
            radii=radii,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        summary += [
            af.text_util.label_and_value_string(
                label=prefix + "concentration",
                value=concentration,
                whitespace=whitespace,
            )
        ]

        radius_at_200 = self.radius_at_200_for_units(
            radii=radii,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        summary += [
            af.text_util.label_value_and_unit_string(
                label=prefix + "radius_at_200x_cosmic_density",
                value=radius_at_200,
                unit=unit_length,
                whitespace=whitespace,
            )
        ]

        mass_at_200 = self.mass_at_200_for_units(
            radii=radii,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        summary += [
            af.text_util.label_value_and_unit_string(
                label=prefix + "mass_at_200x_cosmic_density",
                value=mass_at_200,
                unit=unit_mass,
                whitespace=whitespace,
            )
        ]

        return summary

    @property
    def unit_mass(self):
        return "angular"


class EllipticalGeneralizedNFW(AbstractEllipticalGeneralizedNFW):
    @reshape_returned_array
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def potential_from_grid(
        self, grid, return_in_2d=False, return_binned=False, tabulate_bins=1000
    ):
        """
        Calculate the potential at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        tabulate_bins : int
            The number of bins to tabulate the inner integral of this profile.
        return_in_2d : bool
            If *True*, the returned array is mapped to its unmasked 2D shape, if *False* it is the masked 1D shape.
        return_binned : bool
            If *True*, the returned array which is computed on a sub-grid is binned up to the regular grid dimensions \
            by taking the mean of all sub-gridded values. If *False*, the array is returned on the dimensions of the \
            sub-grid.
        """

        @jit_integrand
        def deflection_integrand(x, kappa_radius, scale_radius, inner_slope):
            return (x + kappa_radius / scale_radius) ** (inner_slope - 3) * (
                (1 - np.sqrt(1 - x ** 2)) / x
            )

        eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size = self.tabulate_integral(
            grid, tabulate_bins
        )

        potential_grid = np.zeros(grid.shape[0])

        deflection_integral = np.zeros((tabulate_bins,))

        for i in range(tabulate_bins):
            eta = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)

            integral = quad(
                deflection_integrand,
                a=0.0,
                b=1.0,
                args=(eta, self.scale_radius, self.inner_slope),
                epsrel=EllipticalGeneralizedNFW.epsrel,
            )[0]

            deflection_integral[i] = (
                (eta / self.scale_radius) ** (2 - self.inner_slope)
            ) * (
                (1.0 / (3 - self.inner_slope))
                * special.hyp2f1(
                    3 - self.inner_slope,
                    3 - self.inner_slope,
                    4 - self.inner_slope,
                    -(eta / self.scale_radius),
                )
                + integral
            )

        for i in range(grid.shape[0]):
            potential_grid[i] = (2.0 * self.kappa_s * self.axis_ratio) * quad(
                self.potential_func,
                a=0.0,
                b=1.0,
                args=(
                    grid[i, 0],
                    grid[i, 1],
                    self.axis_ratio,
                    minimum_log_eta,
                    maximum_log_eta,
                    tabulate_bins,
                    deflection_integral,
                ),
                epsrel=EllipticalGeneralizedNFW.epsrel,
            )[0]

        return potential_grid

    @reshape_returned_grid
    @grids.grid_interpolate
    @geometry_profiles.cache
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(
        self, grid, return_in_2d=False, return_binned=False, tabulate_bins=1000
    ):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        tabulate_bins : int
            The number of bins to tabulate the inner integral of this profile.
        return_in_2d : bool
            If *True*, the returned array is mapped to its unmasked 2D shape, if *False* it is the masked 1D shape.
        return_binned : bool
            If *True*, the returned array which is computed on a sub-grid is binned up to the regular grid dimensions \
            by taking the mean of all sub-gridded values. If *False*, the array is returned on the dimensions of the \
            sub-grid.
        """

        @jit_integrand
        def surface_density_integrand(x, kappa_radius, scale_radius, inner_slope):
            return (
                (3 - inner_slope)
                * (x + kappa_radius / scale_radius) ** (inner_slope - 4)
                * (1 - np.sqrt(1 - x * x))
            )

        def calculate_deflection_component(npow, index):
            deflection_grid = 2.0 * self.kappa_s * self.axis_ratio * grid[:, index]
            deflection_grid *= quad_grid(
                self.deflection_func,
                0.0,
                1.0,
                grid,
                args=(
                    npow,
                    self.axis_ratio,
                    minimum_log_eta,
                    maximum_log_eta,
                    tabulate_bins,
                    surface_density_integral,
                ),
                epsrel=EllipticalGeneralizedNFW.epsrel,
            )[0]

            return deflection_grid

        eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size = self.tabulate_integral(
            grid, tabulate_bins
        )

        surface_density_integral = np.zeros((tabulate_bins,))

        for i in range(tabulate_bins):
            eta = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)

            integral = quad(
                surface_density_integrand,
                a=0.0,
                b=1.0,
                args=(eta, self.scale_radius, self.inner_slope),
                epsrel=EllipticalGeneralizedNFW.epsrel,
            )[0]

            surface_density_integral[i] = (
                (eta / self.scale_radius) ** (1 - self.inner_slope)
            ) * (((1 + eta / self.scale_radius) ** (self.inner_slope - 3)) + integral)

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    def convergence_func(self, radius):
        def integral_y(y, eta):
            return (y + eta) ** (self.inner_slope - 4) * (1 - np.sqrt(1 - y ** 2))

        radius = (1.0 / self.scale_radius) * radius
        integral_y = quad(
            integral_y,
            a=0.0,
            b=1.0,
            args=radius,
            epsrel=EllipticalGeneralizedNFW.epsrel,
        )[0]

        return (
            2.0
            * self.kappa_s
            * (radius ** (1 - self.inner_slope))
            * (
                (1 + radius) ** (self.inner_slope - 3)
                + ((3 - self.inner_slope) * integral_y)
            )
        )

    @staticmethod
    # TODO : Decorator needs to know that potential_integral is 1D array
    #    @jit_integrand
    def potential_func(
        u,
        y,
        x,
        axis_ratio,
        minimum_log_eta,
        maximum_log_eta,
        tabulate_bins,
        potential_integral,
    ):
        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
        i = 1 + int((np.log10(eta_u) - minimum_log_eta) / bin_size)
        r1 = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)
        r2 = r1 * 10.0 ** bin_size
        phi = potential_integral[i] + (
            potential_integral[i + 1] - potential_integral[i]
        ) * (eta_u - r1) / (r2 - r1)
        return eta_u * (phi / u) / (1.0 - (1.0 - axis_ratio ** 2) * u) ** 0.5

    @staticmethod
    # TODO : Decorator needs to know that surface_density_integral is 1D array
    #    @jit_integrand
    def deflection_func(
        u,
        y,
        x,
        npow,
        axis_ratio,
        minimum_log_eta,
        maximum_log_eta,
        tabulate_bins,
        surface_density_integral,
    ):

        eta_u = np.sqrt((u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u)))))
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
        i = 1 + int((np.log10(eta_u) - minimum_log_eta) / bin_size)
        r1 = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)
        r2 = r1 * 10.0 ** bin_size
        kap = surface_density_integral[i] + (
            surface_density_integral[i + 1] - surface_density_integral[i]
        ) * (eta_u - r1) / (r2 - r1)
        return kap / (1.0 - (1.0 - axis_ratio ** 2) * u) ** (npow + 0.5)


class SphericalGeneralizedNFW(EllipticalGeneralizedNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        kappa_s: float = 0.05,
        inner_slope: float = 1.0,
        scale_radius: dim.Length = 1.0,
    ):
        """
        The spherical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        inner_slope : float
            The inner slope of the dark matter halo.
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(SphericalGeneralizedNFW, self).__init__(
            centre=centre,
            axis_ratio=1.0,
            phi=0.0,
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )

    @reshape_returned_grid
    @grids.grid_interpolate
    @geometry_profiles.cache
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = np.multiply(1.0 / self.scale_radius, self.grid_to_grid_radii(grid))

        deflection_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            deflection_grid[i] = np.multiply(
                4.0 * self.kappa_s * self.scale_radius, self.deflection_func_sph(eta[i])
            )

        return self.grid_to_grid_cartesian(grid, deflection_grid)

    @staticmethod
    def deflection_integrand(y, eta, inner_slope):
        return (y + eta) ** (inner_slope - 3) * ((1 - np.sqrt(1 - y ** 2)) / y)

    def deflection_func_sph(self, eta):
        integral_y_2 = quad(
            self.deflection_integrand,
            a=0.0,
            b=1.0,
            args=(eta, self.inner_slope),
            epsrel=1.49e-6,
        )[0]
        return eta ** (2 - self.inner_slope) * (
            (1.0 / (3 - self.inner_slope))
            * special.hyp2f1(
                3 - self.inner_slope, 3 - self.inner_slope, 4 - self.inner_slope, -eta
            )
            + integral_y_2
        )


class SphericalTruncatedNFW(AbstractEllipticalGeneralizedNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: dim.Length = 1.0,
        truncation_radius: dim.Length = 2.0,
    ):
        super(SphericalTruncatedNFW, self).__init__(
            centre=centre,
            axis_ratio=1.0,
            phi=0.0,
            kappa_s=kappa_s,
            inner_slope=1.0,
            scale_radius=scale_radius,
        )

        self.truncation_radius = truncation_radius
        self.tau = self.truncation_radius / self.scale_radius

    def coord_func_k(self, grid_radius):
        return np.log(
            np.divide(
                grid_radius,
                np.sqrt(np.square(grid_radius) + np.square(self.tau)) + self.tau,
            )
        )

    def coord_func_l(self, grid_radius):
        f_r = self.coord_func_f(grid_radius=grid_radius)
        g_r = self.coord_func_g(grid_radius=grid_radius)
        k_r = self.coord_func_k(grid_radius=grid_radius)

        return np.divide(self.tau ** 2.0, (self.tau ** 2.0 + 1.0) ** 2.0) * (
            ((self.tau ** 2.0 + 1.0) * g_r)
            + (2 * f_r)
            - (np.pi / (np.sqrt(self.tau ** 2.0 + grid_radius ** 2.0)))
            + (
                (
                    (self.tau ** 2.0 - 1.0)
                    / (self.tau * (np.sqrt(self.tau ** 2.0 + grid_radius ** 2.0)))
                )
                * k_r
            )
        )

    def coord_func_m(self, grid_radius):
        f_r = self.coord_func_f(grid_radius=grid_radius)
        k_r = self.coord_func_k(grid_radius=grid_radius)

        return (self.tau ** 2.0 / (self.tau ** 2.0 + 1.0) ** 2.0) * (
            ((self.tau ** 2.0 + 2.0 * grid_radius ** 2.0 - 1.0) * f_r)
            + (np.pi * self.tau)
            + ((self.tau ** 2.0 - 1.0) * np.log(self.tau))
            + (
                np.sqrt(grid_radius ** 2.0 + self.tau ** 2.0)
                * (((self.tau ** 2.0 - 1.0) / self.tau) * k_r - np.pi)
            )
        )

    def convergence_func(self, grid_radius):
        grid_radius = ((1.0 / self.scale_radius) * grid_radius) + 0j
        return np.real(2.0 * self.kappa_s * self.coord_func_l(grid_radius=grid_radius))

    def deflection_func_sph(self, grid_radius):
        grid_radius = grid_radius + 0j
        return np.real(self.coord_func_m(grid_radius=grid_radius))

    @reshape_returned_array
    def potential_from_grid(self, grid, return_in_2d=True, return_binned=True):
        return np.zeros((grid.shape[0],))

    @reshape_returned_grid
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def deflections_from_grid(self, grid, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid : grids.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = np.multiply(1.0 / self.scale_radius, self.grid_to_grid_radii(grid))

        deflection_grid = np.multiply(
            (4.0 * self.kappa_s * self.scale_radius / eta),
            self.deflection_func_sph(eta),
        )

        return self.grid_to_grid_cartesian(grid, deflection_grid)

    @dim.convert_units_to_input_units
    def mass_at_truncation_radius(
        self,
        redshift_profile,
        redshift_source,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        mass_at_200 = self.mass_at_200_for_units(
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        return (
            mass_at_200
            * (self.tau ** 2.0 / (self.tau ** 2.0 + 1.0) ** 2.0)
            * (
                ((self.tau ** 2.0 - 1) * np.log(self.tau))
                + (self.tau * np.pi)
                - (self.tau ** 2.0 + 1)
            )
        )

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
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        summary = super().summarize_in_units(
            radii=radii,
            prefix=prefix,
            whitespace=whitespace,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        mass_at_truncation_radius = self.mass_at_truncation_radius(
            radii=radii,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        summary += [
            af.text_util.label_value_and_unit_string(
                label=prefix + "mass_at_truncation_radius",
                value=mass_at_truncation_radius,
                unit=unit_mass,
                whitespace=whitespace,
            )
        ]

        return summary


class SphericalTruncatedNFWChallenge(SphericalTruncatedNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: dim.Length = 1.0,
    ):
        def solve_c(c, de_c):
            """
            Equation need for solving concentration c for a given delta_c
            """
            return 200.0 / 3.0 * (c * c * c / (np.log(1 + c) - c / (1 + c))) - de_c

        kpc_per_arcsec = 6.68549148608755
        scale_radius_kpc = scale_radius * kpc_per_arcsec
        cosmic_average_density = 262.30319684750657
        critical_surface_density = 1940654909.413325
        rho_s = kappa_s * critical_surface_density / scale_radius_kpc
        de_c = rho_s / cosmic_average_density  # delta_c
        concentration = fsolve(solve_c, 10.0, args=(de_c,))[0]
        r200 = concentration * scale_radius_kpc / kpc_per_arcsec  # R_200

        super(SphericalTruncatedNFWChallenge, self).__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            truncation_radius=2.0 * r200,
        )

    @dim.convert_units_to_input_units
    def summarize_in_units(
        self,
        radii,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        summary = super().summarize_in_units(
            radii=radii,
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        return summary


class EllipticalNFW(AbstractEllipticalGeneralizedNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        kappa_s: float = 0.05,
        scale_radius: dim.Length = 1.0,
    ):
        """
        The elliptical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis.
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(EllipticalNFW, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            phi=phi,
            kappa_s=kappa_s,
            inner_slope=1.0,
            scale_radius=scale_radius,
        )

    @staticmethod
    def coord_func(r):
        if r > 1:
            return (1.0 / np.sqrt(r ** 2 - 1)) * np.arctan(np.sqrt(r ** 2 - 1))
        elif r < 1:
            return (1.0 / np.sqrt(1 - r ** 2)) * np.arctanh(np.sqrt(1 - r ** 2))
        elif r == 1:
            return 1

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
            args=(self.axis_ratio, self.kappa_s, self.scale_radius),
            epsrel=1.49e-5,
        )[0]

        return potential_grid

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
            deflection_grid = self.axis_ratio * grid[:, index]
            deflection_grid *= quad_grid(
                self.deflection_func,
                0.0,
                1.0,
                grid,
                args=(npow, self.axis_ratio, self.kappa_s, self.scale_radius),
            )[0]

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotate_grid_from_profile(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    def convergence_func(self, grid_radius):
        grid_radius = (1.0 / self.scale_radius) * grid_radius + 0j
        return np.real(2.0 * self.kappa_s * self.coord_func_g(grid_radius=grid_radius))

    @staticmethod
    def potential_func(u, y, x, axis_ratio, kappa_s, scale_radius):
        eta_u = (1.0 / scale_radius) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u))))
        )

        if eta_u > 1:
            eta_u_2 = (1.0 / np.sqrt(eta_u ** 2 - 1)) * np.arctan(
                np.sqrt(eta_u ** 2 - 1)
            )
        elif eta_u < 1:
            eta_u_2 = (1.0 / np.sqrt(1 - eta_u ** 2)) * np.arctanh(
                np.sqrt(1 - eta_u ** 2)
            )
        else:
            eta_u_2 = 1

        return (
            4.0
            * kappa_s
            * scale_radius
            * (axis_ratio / 2.0)
            * (eta_u / u)
            * ((np.log(eta_u / 2.0) + eta_u_2) / eta_u)
            / ((1 - (1 - axis_ratio ** 2) * u) ** 0.5)
        )

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, kappa_s, scale_radius):
        eta_u = (1.0 / scale_radius) * np.sqrt(
            (u * ((x ** 2) + (y ** 2 / (1 - (1 - axis_ratio ** 2) * u))))
        )

        if eta_u > 1:
            eta_u_2 = (1.0 / np.sqrt(eta_u ** 2 - 1)) * np.arctan(
                np.sqrt(eta_u ** 2 - 1)
            )
        elif eta_u < 1:
            eta_u_2 = (1.0 / np.sqrt(1 - eta_u ** 2)) * np.arctanh(
                np.sqrt(1 - eta_u ** 2)
            )
        else:
            eta_u_2 = 1

        return (
            2.0
            * kappa_s
            * (1 - eta_u_2)
            / (eta_u ** 2 - 1)
            / ((1 - (1 - axis_ratio ** 2) * u) ** (npow + 0.5))
        )


class SphericalNFW(EllipticalNFW):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: dim.Length = 1.0,
    ):
        """
        The spherical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        kappa_s : float
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        scale_radius : float
            The arc-second radius where the average density within this radius is 200 times the critical density of \
            the Universe..
        """

        super(SphericalNFW, self).__init__(
            centre=centre,
            axis_ratio=1.0,
            phi=0.0,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
        )

    # TODO : The 'func' routines require a different input to the elliptical cases, meaning they cannot be overridden.
    # TODO : Should be able to refactor code to deal with this nicely, but will wait until we're clear on numba.

    # TODO : Make this use numpy arithmetic

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
        eta = (1.0 / self.scale_radius) * self.grid_to_grid_radii(grid) + 0j
        return np.real(
            2.0 * self.scale_radius * self.kappa_s * self.potential_func_sph(eta)
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
        eta = np.multiply(1.0 / self.scale_radius, self.grid_to_grid_radii(grid=grid))
        deflection_r = np.multiply(
            4.0 * self.kappa_s * self.scale_radius, self.deflection_func_sph(eta)
        )

        return self.grid_to_grid_cartesian(grid, deflection_r)

    @staticmethod
    def potential_func_sph(eta):
        return ((np.log(eta / 2.0)) ** 2) - (np.arctanh(np.sqrt(1 - eta ** 2))) ** 2

    @staticmethod
    def deflection_func_sph(eta):
        conditional_eta = np.copy(eta)
        conditional_eta[eta > 1] = np.multiply(
            np.divide(1.0, np.sqrt(np.add(np.square(eta[eta > 1]), -1))),
            np.arctan(np.sqrt(np.add(np.square(eta[eta > 1]), -1))),
        )
        conditional_eta[eta < 1] = np.multiply(
            np.divide(1.0, np.sqrt(np.add(1, -np.square(eta[eta < 1])))),
            np.arctanh(np.sqrt(np.add(1, -np.square(eta[eta < 1])))),
        )

        return np.divide(np.add(np.log(np.divide(eta, 2.0)), conditional_eta), eta)
