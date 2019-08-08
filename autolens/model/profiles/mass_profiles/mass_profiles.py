import inspect
from pyquad import quad_grid

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from astropy import cosmology as cosmo
from skimage import measure


import autofit as af
from autolens import dimensions as dim
from autolens import text_util
from autolens.data.array import grids
from autolens.model.profiles import geometry_profiles
from autolens.data.array.util import grid_util

from autolens.data.array.grids import reshape_returned_array, reshape_returned_grid


class MassProfile(object):
    def convergence_func(self, eta):
        raise NotImplementedError("surface_density_func should be overridden")

    def convergence_from_grid(self, grid, return_in_2d=True, return_binned=True):
        pass
        # raise NotImplementedError("surface_density_from_grid should be overridden")

    def potential_from_grid(self, grid, return_in_2d=True, return_binned=True):
        pass
        # raise NotImplementedError("potential_from_grid should be overridden")

    def deflections_from_grid(self, grid, return_in_2d=True, return_binned=True):
        raise NotImplementedError("deflections_from_grid should be overridden")

    def mass_within_circle_in_units(
        self,
        radius: dim.Length,
        redshift_profile=None,
        redshift_source=None,
        unit_mass="solMass",
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        raise NotImplementedError()

    def mass_within_ellipse_in_units(
        self,
        major_axis: dim.Length,
        redshift_profile=None,
        redshift_source=None,
        unit_mass="solMass",
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        raise NotImplementedError()

    def einstein_radius_in_units(
        self, unit_length="arcsec", redshift_profile=None, cosmology=cosmo.Planck15
    ):
        return NotImplementedError()

    def einstein_mass_in_units(
        self,
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        return NotImplementedError()

    def summarize_in_units(
        self,
        radii,
        prefix="",
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        whitespace=80,
        **kwargs
    ):
        return ["Mass Profile = {}\n".format(self.__class__.__name__)]

    @property
    def unit_mass(self):
        return NotImplementedError()


# noinspection PyAbstractClass
class EllipticalMassProfile(geometry_profiles.EllipticalProfile, MassProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
    ):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ellipse's minor-to-major axis ratio (b/a)
        phi : float
            Rotation angle of profile's ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalMassProfile, self).__init__(
            centre=centre, axis_ratio=axis_ratio, phi=phi
        )
        self.axis_ratio = axis_ratio
        self.phi = phi

    @dim.convert_units_to_input_units
    def mass_within_circle_in_units(
        self,
        radius: dim.Length,
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        """ Integrate the mass profiles's convergence profile to compute the total mass within a circle of \
        specified radius. This is centred on the mass profile.

        The following units for mass can be specified and output:

        - Dimensionless angular units (default) - 'solMass'.
        - Solar masses - 'solMass' (multiplies the angular mass by the critical surface mass density).

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        unit_mass : str
            The units the mass is returned in (angular | angular).
        critical_surface_density : float or None
            The critical surface mass density of the strong lens configuration, which converts mass from angulalr \
            units to phsical units (e.g. solar masses).
        """

        critical_surface_density = (
            kwargs["critical_surface_density"]
            if "critical_surface_density" in kwargs
            else None
        )

        mass = dim.Mass(
            value=quad(self.mass_integral, a=0.0, b=radius, args=(1.0,))[0],
            unit_mass=self.unit_mass,
        )

        return mass.convert(
            unit_mass=unit_mass, critical_surface_density=critical_surface_density
        )

    @dim.convert_units_to_input_units
    def mass_within_ellipse_in_units(
        self,
        major_axis: dim.Length,
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        """ Integrate the mass profiles's convergence profile to compute the total angular mass within an ellipse of \
        specified major axis. This is centred on the mass profile.

        The following units for mass can be specified and output:

        - Dimensionless angular units (default) - 'solMass'.
        - Solar masses - 'solMass' (multiplies the angular mass by the critical surface mass density)

        Parameters
        ----------
        major_axis : float
            The major-axis radius of the ellipse.
        unit_mass : str
            The units the mass is returned in (angular | angular).
        critical_surface_density : float or None
            The critical surface mass density of the strong lens configuration, which converts mass from angular \
            units to phsical units (e.g. solar masses).
        """

        critical_surface_density = (
            kwargs["critical_surface_density"]
            if "critical_surface_density" in kwargs
            else None
        )

        mass = dim.Mass(
            value=quad(
                self.mass_integral, a=0.0, b=major_axis, args=(self.axis_ratio,)
            )[0],
            unit_mass=self.unit_mass,
        )

        return mass.convert(
            unit_mass=unit_mass, critical_surface_density=critical_surface_density
        )

    def mass_integral(self, x, axis_ratio):
        """Routine to integrate an elliptical light profiles - set axis ratio to 1 to compute the luminosity within a \
        circle"""
        r = x * axis_ratio
        return 2 * np.pi * r * self.convergence_func(x)

    def density_between_circular_annuli_in_angular_units(
        self,
        inner_annuli_radius: dim.Length,
        outer_annuli_radius: dim.Length,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        """Calculate the mass between two circular annuli and compute the density by dividing by the annuli surface
        area.

        The value returned by the mass integral is dimensionless, therefore the density between annuli is returned in \
        units of inverse radius squared. A conversion factor can be specified to convert this to a physical value \
        (e.g. the critical surface mass density).

        Parameters
        -----------
        inner_annuli_radius : float
            The radius of the inner annulus outside of which the density are estimated.
        outer_annuli_radius : float
            The radius of the outer annulus inside of which the density is estimated.
        """
        annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (
            np.pi * inner_annuli_radius ** 2.0
        )

        outer_mass = self.mass_within_circle_in_units(
            radius=outer_annuli_radius,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            unit_mass=unit_mass,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        inner_mass = self.mass_within_circle_in_units(
            radius=inner_annuli_radius,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            unit_mass=unit_mass,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        return dim.MassOverLength2(
            value=(outer_mass - inner_mass) / annuli_area,
            unit_length=unit_length,
            unit_mass=unit_mass,
        )

    @dim.convert_units_to_input_units
    def average_convergence_of_1_radius_in_units(
        self,
        unit_length="arcsec",
        redshift_profile=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        """The radius a critical curve forms for this mass profile, e.g. where the mean convergence is equal to 1.0.

         In case of ellipitical mass profiles, the 'average' critical curve is used, whereby the convergence is \
         rescaled into a circle using the axis ratio.

         This radius corresponds to the Einstein radius of the mass profile, and is a property of a number of \
         mass profiles below.
         """

        kpc_per_arcsec = (
            kwargs["kpc_per_arcsec"] if "kpc_per_arcsec" in kwargs else None
        )

        def func(radius, redshift_profile, cosmology):
            radius = dim.Length(radius, unit_length=unit_length)
            return (
                self.mass_within_circle_in_units(
                    unit_mass="angular",
                    radius=radius,
                    redshift_profile=redshift_profile,
                    cosmology=cosmology,
                )
                - np.pi * radius ** 2.0
            )

        radius = (
            self.ellipticity_rescale
            * root_scalar(
                func, bracket=[1e-4, 1000.0], args=(redshift_profile, cosmology)
            ).root
        )
        radius = dim.Length(radius, unit_length)
        return radius.convert(unit_length=unit_length, kpc_per_arcsec=kpc_per_arcsec)

    @dim.convert_units_to_input_units
    def einstein_radius_in_units(
        self,
        unit_length="arcsec",
        redshift_profile=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        einstein_radius = self.average_convergence_of_1_radius_in_units(
            unit_length=unit_length,
            redshift_profile=redshift_profile,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        return dim.Length(einstein_radius, unit_length)

    @dim.convert_units_to_input_units
    def einstein_mass_in_units(
        self,
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs
    ):
        einstein_radius = self.einstein_radius_in_units(
            unit_length=self.unit_length,
            redshift_profile=redshift_profile,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        return self.mass_within_circle_in_units(
            radius=einstein_radius,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            cosmology=cosmology,
            kwargs=kwargs,
        )

    @reshape_returned_grid
    def deflections_via_potential_from_grid(
        self, grid, return_in_2d=True, return_binned=True
    ):

        potential_2d = self.potential_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        deflections_y_2d = np.gradient(potential_2d, grid.in_2d[:, 0, 0], axis=0)
        deflections_x_2d = np.gradient(potential_2d, grid.in_2d[0, :, 1], axis=1)

        return np.stack((deflections_y_2d, deflections_x_2d), axis=-1)

    @reshape_returned_array
    def lensing_jacobian_a11_from_grid_and_deflections_2d(
        self, grid, return_in_2d=True, return_binned=True
    ):

        deflections_2d = self.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        return 1.0 - np.gradient(deflections_2d[:, :, 1], grid.in_2d[0, :, 1], axis=1)

    @reshape_returned_array
    def lensing_jacobian_a12_from_grid_and_deflections_2d(
        self, grid, return_in_2d=True, return_binned=True
    ):

        deflections_2d = self.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        return -1.0 * np.gradient(deflections_2d[:, :, 1], grid.in_2d[:, 0, 0], axis=0)

    @reshape_returned_array
    def lensing_jacobian_a21_from_grid_and_deflections_2d(
        self, grid, return_in_2d=True, return_binned=True
    ):

        deflections_2d = self.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        return -1.0 * np.gradient(deflections_2d[:, :, 0], grid.in_2d[0, :, 1], axis=1)

    @reshape_returned_array
    def lensing_jacobian_a22_from_grid_and_deflections_2d(
        self, grid, return_in_2d=True, return_binned=True
    ):

        deflections_2d = self.deflections_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        return 1 - np.gradient(deflections_2d[:, :, 0], grid.in_2d[:, 0, 0], axis=0)

    def lensing_jacobian_from_grid(self, grid, return_in_2d=True, return_binned=True):

        a11 = self.lensing_jacobian_a11_from_grid_and_deflections_2d(
            grid=grid, return_in_2d=return_in_2d, return_binned=return_binned
        )

        a12 = self.lensing_jacobian_a12_from_grid_and_deflections_2d(
            grid=grid, return_in_2d=return_in_2d, return_binned=return_binned
        )

        a21 = self.lensing_jacobian_a21_from_grid_and_deflections_2d(
            grid=grid, return_in_2d=return_in_2d, return_binned=return_binned
        )

        a22 = self.lensing_jacobian_a22_from_grid_and_deflections_2d(
            grid=grid, return_in_2d=return_in_2d, return_binned=return_binned
        )

        return np.array([[a11, a12], [a21, a22]])

    @reshape_returned_array
    def convergence_from_jacobian(self, grid, return_in_2d=True, return_binned=True):

        jacobian = self.lensing_jacobian_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        convergence = 1 - 0.5 * (jacobian[0, 0] + jacobian[1, 1])

        return convergence

    @reshape_returned_array
    def shear_from_jacobian(self, grid, return_in_2d=True, return_binned=True):

        jacobian = self.lensing_jacobian_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        gamma_1 = 0.5 * (jacobian[1, 1] - jacobian[0, 0])
        gamma_2 = -0.5 * (jacobian[0, 1] + jacobian[1, 0])

        return (gamma_1 ** 2 + gamma_2 ** 2) ** 0.5

    @reshape_returned_array
    def tangential_eigen_value_from_shear_and_convergence(
        self, grid, return_in_2d=True, return_binned=True
    ):

        convergence = self.convergence_from_jacobian(
            grid=grid, return_in_2d=False, return_binned=False
        )

        shear = self.shear_from_jacobian(
            grid=grid, return_in_2d=False, return_binned=False
        )

        return 1 - convergence - shear

    @reshape_returned_array
    def radial_eigen_value_from_shear_and_convergence(
        self, grid, return_in_2d=True, return_binned=True
    ):

        convergence = self.convergence_from_jacobian(
            grid=grid, return_in_2d=False, return_binned=False
        )

        shear = self.shear_from_jacobian(
            grid=grid, return_in_2d=False, return_binned=False
        )

        return 1 - convergence + shear

    @reshape_returned_array
    def magnification_from_grid(self, grid, return_in_2d=True, return_binned=True):

        jacobian = self.lensing_jacobian_from_grid(
            grid=grid, return_in_2d=False, return_binned=False
        )

        det_jacobian = jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]

        return 1 / det_jacobian

    def tangential_critical_curve_from_grid(self, grid):

        lambda_tan_2d = self.tangential_eigen_value_from_shear_and_convergence(
            grid=grid, return_in_2d=True, return_binned=False
        )

        tan_critical_curve_indices = measure.find_contours(lambda_tan_2d, 0)

        return grid_util.grid_pixels_1d_to_grid_arcsec_1d(
            grid_pixels_1d=tan_critical_curve_indices[0],
            shape=lambda_tan_2d.shape,
            pixel_scales=(
                grid.pixel_scale / grid.sub_grid_size,
                grid.pixel_scale / grid.sub_grid_size,
            ),
            origin=grid.mask.origin,
        )

    def tangential_caustic_from_grid(self, grid):

        tangential_critical_curve = self.tangential_critical_curve_from_grid(grid=grid)

        deflections_1d = self.deflections_from_grid(
            grid=tangential_critical_curve, return_in_2d=False, return_binned=False
        )

        return tangential_critical_curve - deflections_1d

    def radial_critical_curve_from_grid(self, grid):

        lambda_rad_2d = self.radial_eigen_value_from_shear_and_convergence(
            grid=grid, return_in_2d=False, return_binned=False
        )

        radial_critical_curve_indices = measure.find_contours(lambda_rad_2d, 0)

        ##rad_critical_curve = np.fliplr(radial_critical_curve_indices[0])

        ## fliping x, y coordinates may or may not be necessary, appears to visualise the same either way
        ## reg grid unit test works with this fix, sub grid still doesn't like it
        ## may be an isuue with where the marching squares algorithm starts rathet than x, y flip

        return grid_util.grid_pixels_1d_to_grid_arcsec_1d(
            grid_pixels_1d=radial_critical_curve_indices[0],
            shape=lambda_rad_2d.shape,
            pixel_scales=(
                grid.pixel_scale / grid.sub_grid_size,
                grid.pixel_scale / grid.sub_grid_size,
            ),
            origin=grid.mask.origin,
        )

    def radial_caustic_from_grid(self, grid):

        radial_critical_curve = self.radial_critical_curve_from_grid(grid=grid)

        deflections_1d = self.deflections_from_grid(
            grid=radial_critical_curve, return_in_2d=False, return_binned=False
        )

        return radial_critical_curve - deflections_1d

    def critical_curves_from_grid(self, grid):

        magnification_2d = self.magnification_from_grid(
            grid=grid, return_in_2d=True, return_binned=False
        )

        inverse_magnification_2d = 1 / magnification_2d

        critical_curves_indices = measure.find_contours(inverse_magnification_2d, 0)

        no_critical_curves = len(critical_curves_indices)
        contours = []
        critical_curves = []

        for jj in np.arange(no_critical_curves):

            contours.append(critical_curves_indices[jj])
            contour_x, contour_y = contours[jj].T
            pixel_coord = np.stack((contour_x, contour_y), axis=-1)

            critical_curve = grid_util.grid_pixels_1d_to_grid_arcsec_1d(
                grid_pixels_1d=pixel_coord,
                shape=magnification_2d.shape,
                pixel_scales=(
                    grid.pixel_scale / grid.sub_grid_size,
                    grid.pixel_scale / grid.sub_grid_size,
                ),
                origin=grid.mask.origin,
            )

            critical_curves.append(critical_curve)

        return critical_curves

    def caustics_from_grid(self, grid):

        caustics = []

        critical_curves = self.critical_curves_from_grid(grid=grid)

        for i in range(len(critical_curves)):

            critical_curve = critical_curves[i]

            deflections_1d = self.deflections_from_grid(
                grid=critical_curve, return_in_2d=False, return_binned=False
            )

            caustic = critical_curve - deflections_1d

            caustics.append(caustic)

        return caustics

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
            prefix="",
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        einstein_radius = self.einstein_radius_in_units(
            unit_length=unit_length,
            redshift_profile=redshift_profile,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        summary += [
            af.text_util.label_value_and_unit_string(
                label=prefix + "einstein_radius",
                value=einstein_radius,
                unit=unit_length,
                whitespace=whitespace,
            )
        ]

        einstein_mass = self.einstein_mass_in_units(
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        summary += [
            af.text_util.label_value_and_unit_string(
                label=prefix + "einstein_mass",
                value=einstein_mass,
                unit=unit_mass,
                whitespace=whitespace,
            )
        ]

        for radius in radii:
            mass = self.mass_within_circle_in_units(
                unit_mass=unit_mass,
                radius=radius,
                redshift_profile=redshift_profile,
                redshift_source=redshift_source,
                cosmology=cosmology,
                kwargs=kwargs,
            )

            summary += [
                text_util.within_radius_label_value_and_unit_string(
                    prefix=prefix + "mass",
                    radius=radius,
                    unit_length=unit_length,
                    value=mass,
                    unit_value=unit_mass,
                    whitespace=whitespace,
                )
            ]

        return summary

    @property
    def ellipticity_rescale(self):
        return NotImplementedError()
