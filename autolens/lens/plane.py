import numpy as np
from astropy import cosmology as cosmo

import autofit as af
from autolens import exc, dimensions as dim
from autolens.data.array import scaled_array
from autolens.lens.util import lens_util
from autolens.model import cosmology_util

from autolens.data.array.grids import (
    reshape_returned_array,
    reshape_returned_array_blurring,
    reshape_returned_grid,
)


class AbstractPlane(object):
    def __init__(self, redshift, galaxies=None, cosmology=cosmo.Planck15):
        """An abstract plane which represents a set of galaxies that are close to one another in redshift-space.

        From an abstract plane, cosmological quantities like angular diameter distances can be computed. If the  \
        redshift of the plane is input as *None*, quantities that depend on a redshift are returned as *None*.

        Parameters
        -----------
        redshift : float or None
            The redshift of the plane.
        galaxies : [Galaxy]
            The list of galaxies in this plane.
        cosmology : astropy.cosmology
            The cosmology associated with the plane, used to convert arc-second coordinates to physical values.
        """

        self.redshift = redshift
        self.galaxies = galaxies
        self.cosmology = cosmology

    @property
    def galaxy_redshifts(self):
        return [galaxy.redshift for galaxy in self.galaxies]

    @property
    def arcsec_per_kpc(self):
        return cosmology_util.arcsec_per_kpc_from_redshift_and_cosmology(
            redshift=self.redshift, cosmology=self.cosmology
        )

    @property
    def kpc_per_arcsec(self):
        return 1.0 / self.arcsec_per_kpc

    def angular_diameter_distance_to_earth_in_units(self, unit_length="arcsec"):
        return cosmology_util.angular_diameter_distance_to_earth_from_redshift_and_cosmology(
            redshift=self.redshift, cosmology=self.cosmology, unit_length=unit_length
        )

    def cosmic_average_density_in_units(
        self, unit_length="arcsec", unit_mass="solMass"
    ):
        return cosmology_util.cosmic_average_density_from_redshift_and_cosmology(
            redshift=self.redshift,
            cosmology=self.cosmology,
            unit_length=unit_length,
            unit_mass=unit_mass,
        )

    @property
    def has_light_profile(self):
        return any(list(map(lambda galaxy: galaxy.has_light_profile, self.galaxies)))

    @property
    def has_mass_profile(self):
        return any(list(map(lambda galaxy: galaxy.has_mass_profile, self.galaxies)))

    @property
    def has_pixelization(self):
        return any(list(map(lambda galaxy: galaxy.has_pixelization, self.galaxies)))

    @property
    def has_regularization(self):
        return any(list(map(lambda galaxy: galaxy.has_regularization, self.galaxies)))

    @property
    def regularization(self):

        galaxies_with_regularization = list(
            filter(lambda galaxy: galaxy.has_regularization, self.galaxies)
        )

        if len(galaxies_with_regularization) == 0:
            return None
        if len(galaxies_with_regularization) == 1:
            return galaxies_with_regularization[0].regularization
        elif len(galaxies_with_regularization) > 1:
            raise exc.PixelizationException(
                "The number of galaxies with regularizations in one plane is above 1"
            )

    @property
    def has_hyper_galaxy(self):
        return any(list(map(lambda galaxy: galaxy.has_hyper_galaxy, self.galaxies)))

    @property
    def contribution_maps_1d_of_galaxies(self):

        contribution_maps_1d = []

        for galaxy in self.galaxies:

            if galaxy.hyper_galaxy is not None:

                contribution_map = galaxy.hyper_galaxy.contribution_map_from_hyper_images(
                    hyper_model_image=galaxy.hyper_model_image_1d,
                    hyper_galaxy_image=galaxy.hyper_galaxy_image_1d,
                )

                contribution_maps_1d.append(contribution_map)

            else:

                contribution_maps_1d.append(None)

        return contribution_maps_1d

    @property
    def centres_of_galaxy_mass_profiles(self):

        galaxies_with_mass_profiles = [
            galaxy for galaxy in self.galaxies if galaxy.has_mass_profile
        ]

        mass_profile_centres = [[] for _ in range(len(galaxies_with_mass_profiles))]

        for galaxy_index, galaxy in enumerate(galaxies_with_mass_profiles):
            mass_profile_centres[galaxy_index] = [
                profile.centre for profile in galaxy.mass_profiles
            ]
        return mass_profile_centres

    @property
    def axis_ratios_of_galaxy_mass_profiles(self):
        galaxies_with_mass_profiles = [
            galaxy for galaxy in self.galaxies if galaxy.has_mass_profile
        ]

        mass_profile_axis_ratios = [[] for _ in range(len(galaxies_with_mass_profiles))]

        for galaxy_index, galaxy in enumerate(galaxies_with_mass_profiles):
            mass_profile_axis_ratios[galaxy_index] = [
                profile.axis_ratio for profile in galaxy.mass_profiles
            ]
        return mass_profile_axis_ratios

    @property
    def phis_of_galaxy_mass_profiles(self):

        galaxies_with_mass_profiles = [
            galaxy for galaxy in self.galaxies if galaxy.has_mass_profile
        ]

        mass_profile_phis = [[] for _ in range(len(galaxies_with_mass_profiles))]

        for galaxy_index, galaxy in enumerate(galaxies_with_mass_profiles):
            mass_profile_phis[galaxy_index] = [
                profile.phi for profile in galaxy.mass_profiles
            ]
        return mass_profile_phis

    def luminosities_of_galaxies_within_circles_in_units(
        self, radius: dim.Length, unit_luminosity="eps", exposure_time=None
    ):
        """Compute the total luminosity of all galaxies in this plane within a circle of specified radius.

        See *galaxy.light_within_circle* and *light_profiles.light_within_circle* for details \
        of how this is performed.

        Parameters
        ----------
        radius : float
            The radius of the circle to compute the dimensionless mass within.
        unit_luminosity : str
            The units the luminosity is returned in (eps | counts).
        exposure_time : float
            The exposure time of the observation, which converts luminosity from electrons per second units to counts.
        """
        return list(
            map(
                lambda galaxy: galaxy.luminosity_within_circle_in_units(
                    radius=radius,
                    unit_luminosity=unit_luminosity,
                    exposure_time=exposure_time,
                    cosmology=self.cosmology,
                ),
                self.galaxies,
            )
        )

    def luminosities_of_galaxies_within_ellipses_in_units(
        self, major_axis: dim.Length, unit_luminosity="eps", exposure_time=None
    ):
        """
        Compute the total luminosity of all galaxies in this plane within a ellipse of specified major-axis.

        The value returned by this integral is dimensionless, and a conversion factor can be specified to convert it \
        to a physical value (e.g. the photometric zeropoint).

        See *galaxy.light_within_ellipse* and *light_profiles.light_within_ellipse* for details
        of how this is performed.

        Parameters
        ----------
        major_axis : float
            The major-axis radius of the ellipse.
        unit_luminosity : str
            The units the luminosity is returned in (eps | counts).
        exposure_time : float
            The exposure time of the observation, which converts luminosity from electrons per second units to counts.
        """
        return list(
            map(
                lambda galaxy: galaxy.luminosity_within_ellipse_in_units(
                    major_axis=major_axis,
                    unit_luminosity=unit_luminosity,
                    exposure_time=exposure_time,
                    cosmology=self.cosmology,
                ),
                self.galaxies,
            )
        )

    def masses_of_galaxies_within_circles_in_units(
        self, radius: dim.Length, unit_mass="solMass", redshift_source=None
    ):
        """Compute the total mass of all galaxies in this plane within a circle of specified radius.

        See *galaxy.angular_mass_within_circle* and *mass_profiles.angular_mass_within_circle* for details
        of how this is performed.

        Parameters
        ----------
        redshift_source
        radius : float
            The radius of the circle to compute the dimensionless mass within.
        unit_mass : str
            The units the mass is returned in (angular | solMass).

        """
        return list(
            map(
                lambda galaxy: galaxy.mass_within_circle_in_units(
                    radius=radius,
                    unit_mass=unit_mass,
                    redshift_source=redshift_source,
                    cosmology=self.cosmology,
                ),
                self.galaxies,
            )
        )

    def masses_of_galaxies_within_ellipses_in_units(
        self, major_axis: dim.Length, unit_mass="solMass", redshift_source=None
    ):
        """Compute the total mass of all galaxies in this plane within a ellipse of specified major-axis.

        See *galaxy.angular_mass_within_ellipse* and *mass_profiles.angular_mass_within_ellipse* for details \
        of how this is performed.

        Parameters
        ----------
        redshift_source
        unit_mass
        major_axis : float
            The major-axis radius of the ellipse.

        """
        return list(
            map(
                lambda galaxy: galaxy.mass_within_ellipse_in_units(
                    major_axis=major_axis,
                    unit_mass=unit_mass,
                    redshift_source=redshift_source,
                    cosmology=self.cosmology,
                ),
                self.galaxies,
            )
        )

    def einstein_radius_in_units(self, unit_length="arcsec"):

        if self.has_mass_profile:
            return sum(
                filter(
                    None,
                    list(
                        map(
                            lambda galaxy: galaxy.einstein_radius_in_units(
                                unit_length=unit_length, cosmology=self.cosmology
                            ),
                            self.galaxies,
                        )
                    ),
                )
            )

    def einstein_mass_in_units(self, unit_mass="solMass", redshift_source=None):

        if self.has_mass_profile:
            return sum(
                filter(
                    None,
                    list(
                        map(
                            lambda galaxy: galaxy.einstein_mass_in_units(
                                unit_mass=unit_mass,
                                redshift_source=redshift_source,
                                cosmology=self.cosmology,
                            ),
                            self.galaxies,
                        )
                    ),
                )
            )

    # noinspection PyUnusedLocal
    def summarize_in_units(
        self,
        radii,
        whitespace=80,
        unit_length="arcsec",
        unit_luminosity="eps",
        unit_mass="solMass",
        redshift_source=None,
        **kwargs
    ):

        summary = ["Plane\n"]
        prefix_plane = ""

        summary += [
            af.text_util.label_and_value_string(
                label=prefix_plane + "redshift",
                value=self.redshift,
                whitespace=whitespace,
                format_string="{:.2f}",
            )
        ]

        summary += [
            af.text_util.label_and_value_string(
                label=prefix_plane + "kpc_per_arcsec",
                value=self.kpc_per_arcsec,
                whitespace=whitespace,
                format_string="{:.2f}",
            )
        ]

        angular_diameter_distance_to_earth = self.angular_diameter_distance_to_earth_in_units(
            unit_length=unit_length
        )

        summary += [
            af.text_util.label_and_value_string(
                label=prefix_plane + "angular_diameter_distance_to_earth",
                value=angular_diameter_distance_to_earth,
                whitespace=whitespace,
                format_string="{:.2f}",
            )
        ]

        for galaxy in self.galaxies:
            summary += ["\n"]
            summary += galaxy.summarize_in_units(
                radii=radii,
                whitespace=whitespace,
                unit_length=unit_length,
                unit_luminosity=unit_luminosity,
                unit_mass=unit_mass,
                redshift_source=redshift_source,
                cosmology=self.cosmology,
            )

        return summary


class AbstractGriddedPlane(AbstractPlane):
    def __init__(
        self,
        redshift,
        grid_stack,
        border,
        compute_deflections,
        galaxies=None,
        cosmology=cosmo.Planck15,
    ):
        """An abstract plane which represents a set of galaxies that are close to one another in redshift-space and \
        have an associated grid on which lensing calcuations are performed.

        From an abstract plane grid, the surface-density, potential and deflection angles of the galaxies can be \
        computed.

        Parameters
        -----------
        redshift : float or None
            The redshift of the plane.
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        grid_stack : masks.GridStack
            The stack of grid_stacks of (y,x) arc-second coordinates of this plane.
        border : masks.RegularGridBorder
            The borders of the regular-grid, which is used to relocate demagnified traced regular-pixel to the \
            source-plane borders.
        compute_deflections : bool
            If true, the deflection-angles of this plane's coordinates are calculated use its galaxy's mass-profiles.
        cosmology : astropy.cosmology
            The cosmology associated with the plane, used to convert arc-second coordinates to physical values.
        """

        super(AbstractGriddedPlane, self).__init__(
            redshift=redshift, galaxies=galaxies, cosmology=cosmology
        )

        self.grid_stack = grid_stack
        self.border = border

        if compute_deflections:

            def calculate_deflections(grid):

                if galaxies:
                    return sum(
                        map(
                            lambda galaxy: galaxy.deflections_from_grid(
                                grid, return_in_2d=False, return_binned=False
                            ),
                            galaxies,
                        )
                    )
                else:
                    return np.full((grid.shape[0], 2), 0.0)

            self.deflections_stack = grid_stack.apply_function(calculate_deflections)

        else:

            self.deflections_stack = None

    def trace_grid_stack_to_next_plane(self):
        """Trace this plane's grid_stacks to the next plane, using its deflection angles."""

        def minus(grid, deflections):
            return grid - deflections

        return self.grid_stack.map_function(minus, self.deflections_stack)

    @reshape_returned_array
    def profile_image_plane_image(self, return_in_2d=True, return_binned=True):
        """Compute the profile-image plane image of the list of galaxies of the plane's sub-grid, by summing the
        individual images of each galaxy's light profile.

        The image is calculated on the sub-grid and binned-up to the original regular grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is *True*.

        If the plane has no galaxies (or no galaxies have mass profiles) an array of all zeros the shape of the plane's
        sub-grid is returned.

        Parameters
        -----------
        return_in_2d : bool
            If *True*, the returned array is mapped to its unmasked 2D shape, if *False* it is the masked 1D shape.
        return_binned : bool
            If *True*, the returned array which is computed on a sub-grid is binned up to the regular grid dimensions \
            by taking the mean of all sub-gridded values. If *False*, the array is returned on the dimensions of the \
            sub-grid.
        """
        if self.galaxies:
            return sum(
                map(
                    lambda g: g.intensities_from_grid(
                        grid=self.grid_stack.sub,
                        return_in_2d=False,
                        return_binned=False,
                    ),
                    self.galaxies,
                )
            )
        else:
            return np.full((self.grid_stack.sub.shape[0]), 0.0)

    def profile_image_plane_image_of_galaxies(
        self, return_in_2d=True, return_binned=True
    ):
        return list(
            map(
                lambda galaxy: self.profile_image_plane_image_of_galaxy(
                    galaxy=galaxy,
                    return_in_2d=return_in_2d,
                    return_binned=return_binned,
                ),
                self.galaxies,
            )
        )

    def profile_image_plane_image_of_galaxy(
        self, galaxy, return_in_2d=True, return_binned=True
    ):
        return galaxy.intensities_from_grid(
            grid=self.grid_stack.sub,
            return_in_2d=return_in_2d,
            return_binned=return_binned,
        )

    @reshape_returned_array_blurring
    def profile_image_plane_blurring_image(self, return_in_2d=True):
        """Compute the profile-image plane blurring image of the list of galaxies of the plane's sub-grid, by summing \
        the individual blurring images of each galaxy's light profile.

        The blurring image is calculated on the blurring grid, which is not subgrided.

        If the plane has no galaxies (or no galaxies have mass profiles) an array of all zeros the shape of the plane's
        blurring grid is returned.

        Parameters
        -----------
        return_in_2d : bool
            If *True*, the returned array is mapped to its unmasked 2D shape, if *False* it is the masked 1D shape.
        """
        if self.galaxies:
            return sum(
                map(
                    lambda g: g.intensities_from_grid(
                        grid=self.grid_stack.blurring,
                        return_in_2d=False,
                        return_binned=False,
                    ),
                    self.galaxies,
                )
            )
        else:
            return np.full((self.grid_stack.bluring.shape[0]), 0.0)

    def profile_image_plane_blurring_image_of_galaxies(self, return_in_2d=False):
        return list(
            map(
                lambda galaxy: self.profile_image_plane_blurring_image_of_galaxy(
                    galaxy=galaxy, return_in_2d=return_in_2d
                ),
                self.galaxies,
            )
        )

    def profile_image_plane_blurring_image_of_galaxy(self, galaxy, return_in_2d=False):
        return galaxy.intensities_from_grid(
            grid=self.grid_stack.blurring,
            return_in_2d=return_in_2d,
            return_binned=False,
        )

    @reshape_returned_array
    def convergence(self, return_in_2d=True, return_binned=True):
        """Compute the convergence of the list of galaxies of the plane's sub-grid, by summing the individual convergences \
        of each galaxy's mass profile.

        The convergence is calculated on the sub-grid and binned-up to the original regular grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is *True*.

        If the plane has no galaxies (or no galaxies have mass profiles) an array of all zeros the shape of the plane's
        sub-grid is returned.

        Parameters
        -----------
        grid : RegularGrid
            The grid (regular or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
            potential is calculated on.
        galaxies : [galaxy.Galaxy]
            The galaxies whose mass profiles are used to compute the surface densities.
        return_in_2d : bool
            If *True*, the returned array is mapped to its unmasked 2D shape, if *False* it is the masked 1D shape.
        return_binned : bool
            If *True*, the returned array which is computed on a sub-grid is binned up to the regular grid dimensions \
            by taking the mean of all sub-gridded values. If *False*, the array is returned on the dimensions of the \
            sub-grid.
        """
        if self.galaxies:
            return sum(
                map(
                    lambda g: g.convergence_from_grid(
                        grid=self.grid_stack.sub.unlensed_grid_1d,
                        return_in_2d=False,
                        return_binned=False,
                    ),
                    self.galaxies,
                )
            )
        else:
            return np.full((self.grid_stack.sub.shape[0]), 0.0)

    @reshape_returned_array
    def potential(self, return_in_2d=True, return_binned=True):
        """Compute the potential of the list of galaxies of the plane's sub-grid, by summing the individual potentials \
        of each galaxy's mass profile.

        The potential is calculated on the sub-grid and binned-up to the original regular grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is *True*.

        If the plane has no galaxies (or no galaxies have mass profiles) an array of all zeros the shape of the plane's
        sub-grid is returned.

        Parameters
        -----------
        grid : RegularGrid
            The grid (regular or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
            potential is calculated on.
        galaxies : [galaxy.Galaxy]
            The galaxies whose mass profiles are used to compute the surface densities.
        return_in_2d : bool
            If *True*, the returned array is mapped to its unmasked 2D shape, if *False* it is the masked 1D shape.
        return_binned : bool
            If *True*, the returned array which is computed on a sub-grid is binned up to the regular grid dimensions \
            by taking the mean of all sub-gridded values. If *False*, the array is returned on the dimensions of the \
            sub-grid.
        """
        if self.galaxies:
            return sum(
                map(
                    lambda g: g.potential_from_grid(
                        grid=self.grid_stack.sub.unlensed_grid_1d,
                        return_in_2d=False,
                        return_binned=False,
                    ),
                    self.galaxies,
                )
            )
        else:
            return np.full((self.grid_stack.sub.shape[0]), 0.0)

    @reshape_returned_array
    def deflections_y(self, return_in_2d=True, return_binned=True):
        return self.deflections(return_in_2d=False, return_binned=False)[:, 0]

    @reshape_returned_array
    def deflections_x(self, return_in_2d=True, return_binned=True):
        return self.deflections(return_in_2d=False, return_binned=False)[:, 1]

    @reshape_returned_grid
    def deflections(self, return_in_2d=True, return_binned=True):
        if self.galaxies:
            return sum(
                map(
                    lambda g: g.deflections_from_grid(
                        grid=self.grid_stack.sub.unlensed_grid_1d,
                        return_in_2d=False,
                        return_binned=False,
                    ),
                    self.galaxies,
                )
            )
        else:
            return np.full((self.grid_stack.sub.shape[0], 2), 0.0)

    @property
    def plane_image(self):
        return lens_util.plane_image_of_galaxies_from_grid(
            shape=self.grid_stack.regular.mask.shape,
            grid=self.grid_stack.regular,
            galaxies=self.galaxies,
        )

    @property
    def mapper(self):

        galaxies_with_pixelization = list(
            filter(lambda galaxy: galaxy.has_pixelization, self.galaxies)
        )

        if len(galaxies_with_pixelization) == 0:
            return None
        if len(galaxies_with_pixelization) == 1:

            pixelization = galaxies_with_pixelization[0].pixelization

            return pixelization.mapper_from_grid_stack_and_border(
                grid_stack=self.grid_stack,
                border=self.border,
                hyper_image=galaxies_with_pixelization[0].hyper_galaxy_image_1d,
            )

        elif len(galaxies_with_pixelization) > 1:
            raise exc.PixelizationException(
                "The number of galaxies with pixelizations in one plane is above 1"
            )

    @property
    def yticks(self):
        """Compute the yticks labels of this grid_stack, used for plotting the y-axis ticks when visualizing an image \
        """
        return np.linspace(
            np.amin(self.grid_stack.regular[:, 0]),
            np.amax(self.grid_stack.regular[:, 0]),
            4,
        )

    @property
    def xticks(self):
        """Compute the xticks labels of this grid_stack, used for plotting the x-axis ticks when visualizing an \
        image"""
        return np.linspace(
            np.amin(self.grid_stack.regular[:, 1]),
            np.amax(self.grid_stack.regular[:, 1]),
            4,
        )


class AbstractDataPlane(AbstractGriddedPlane):
    def blurred_profile_image_plane_image_1d_from_convolver_image(
        self, convolver_image
    ):

        image_array = self.profile_image_plane_image(
            return_in_2d=False, return_binned=True
        )
        blurring_array = self.profile_image_plane_blurring_image(return_in_2d=False)

        return convolver_image.convolve_image(
            image_array=image_array, blurring_array=blurring_array
        )

    def blurred_profile_image_plane_images_1d_of_galaxies_from_convolver_image(
        self, convolver_image
    ):

        return list(
            map(
                lambda profile_image_plane_image_1d, profile_image_plane_blurring_image_1d: convolver_image.convolve_image(
                    image_array=profile_image_plane_image_1d,
                    blurring_array=profile_image_plane_blurring_image_1d,
                ),
                self.profile_image_plane_image_of_galaxies(
                    return_in_2d=False, return_binned=True
                ),
                self.profile_image_plane_blurring_image_of_galaxies(return_in_2d=False),
            )
        )

    def hyper_noise_map_1d_from_noise_map_1d(self, noise_map_1d):
        hyper_noise_maps_1d = self.hyper_noise_maps_1d_of_galaxies_from_noise_map_1d(
            noise_map_1d=noise_map_1d
        )
        hyper_noise_maps_1d = [
            hyper_noise_map
            for hyper_noise_map in hyper_noise_maps_1d
            if hyper_noise_map is not None
        ]
        return sum(hyper_noise_maps_1d)

    def hyper_noise_maps_1d_of_galaxies_from_noise_map_1d(self, noise_map_1d):
        """For a contribution map and noise-map, use the model hyper galaxies to compute a scaled noise-map.

        Parameters
        -----------
        noise_map_1d : ccd.NoiseMap or ndarray
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        """
        hyper_noise_maps_1d = []

        for galaxy in self.galaxies:
            if galaxy.hyper_galaxy is not None:

                hyper_noise_map_1d = galaxy.hyper_galaxy.hyper_noise_map_from_hyper_images_and_noise_map(
                    noise_map=noise_map_1d,
                    hyper_model_image=galaxy.hyper_model_image_1d,
                    hyper_galaxy_image=galaxy.hyper_galaxy_image_1d,
                )

                hyper_noise_maps_1d.append(hyper_noise_map_1d)

            else:

                hyper_noise_maps_1d.append(None)

        return hyper_noise_maps_1d


class Plane(AbstractDataPlane):
    def __init__(
        self,
        galaxies,
        grid_stack,
        redshift=None,
        border=None,
        compute_deflections=True,
        cosmology=cosmo.Planck15,
    ):
        """A plane of galaxies where all galaxies are at the same redshift.

        Parameters
        -----------
        redshift : float or None
            The redshift of the plane.
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        grid_stack : masks.GridStack
            The stack of grid_stacks of (y,x) arc-second coordinates of this plane.
        border : masks.RegularGridBorder
            The borders of the regular-grid, which is used to relocate demagnified traced regular-pixel to the \
            source-plane borders.
        compute_deflections : bool
            If true, the deflection-angles of this plane's coordinates are calculated use its galaxy's mass-profiles.
        cosmology : astropy.cosmology
            The cosmology associated with the plane, used to convert arc-second coordinates to physical values.
        """

        if redshift is None:

            if not galaxies:
                raise exc.RayTracingException(
                    "A redshift and no galaxies were input to a Plane. A redshift for the Plane therefore cannot be"
                    "determined"
                )
            elif not all(
                [galaxies[0].redshift == galaxy.redshift for galaxy in galaxies]
            ):
                raise exc.RayTracingException(
                    "A redshift and two or more galaxies with different redshifts were input to a Plane. A unique "
                    "Redshift for the Plane therefore cannot be determined"
                )
            else:
                redshift = galaxies[0].redshift

        super(Plane, self).__init__(
            redshift=redshift,
            galaxies=galaxies,
            grid_stack=grid_stack,
            border=border,
            compute_deflections=compute_deflections,
            cosmology=cosmology,
        )


class PlanePositions(object):
    def __init__(
        self, redshift, galaxies, positions, compute_deflections=True, cosmology=None
    ):
        """A plane represents a set of galaxies at a given redshift in a ray-tracer_normal and the positions of image-plane \
        coordinates which mappers close to one another in the source-plane.

        Parameters
        -----------
        galaxies : [Galaxy]
            The list of lens galaxies in this plane.
        positions : [[[]]]
            The (y,x) arc-second coordinates of image-plane pixels which (are expected to) mappers to the same
            location(s) in the final source-plane.
        compute_deflections : bool
            If true, the deflection-angles of this plane's coordinates are calculated use its galaxy's mass-profiles.
        """

        self.redshift = redshift
        self.galaxies = galaxies
        self.positions = positions

        if compute_deflections:

            def calculate_deflections(pos):
                return sum(
                    map(lambda galaxy: galaxy.deflections_from_grid(pos), galaxies)
                )

            self.deflections = list(
                map(lambda pos: calculate_deflections(pos), self.positions)
            )

        self.cosmology = cosmology

    def trace_to_next_plane(self):
        """Trace the positions to the next plane."""
        return list(
            map(
                lambda positions, deflections: np.subtract(positions, deflections),
                self.positions,
                self.deflections,
            )
        )


class PlaneImage(scaled_array.ScaledRectangularPixelArray):
    def __init__(self, array, pixel_scales, grid, origin=(0.0, 0.0)):
        self.grid = grid
        super(PlaneImage, self).__init__(
            array=array, pixel_scales=pixel_scales, origin=origin
        )
