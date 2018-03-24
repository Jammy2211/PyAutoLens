from auto_lens.profiles import geometry_profiles

import numpy as np

class RayTrace(object):
    """The ray-tracing calculations, defined by the image_grid and source planes of every galaxy in this lensing system.
    These are computed in order of ascending galaxy redshift.

    The ray-tracing calculations between every plane is determined using the angular diameter distances between each \
    set of galaxies.

    This is used to perform all ray-tracing calculations and for converting dimensionless measurements (e.g. \
    arc-seconds, mass) to physical units.

    Parameters
    ----------
    galaxies : [Galaxy]
        The list of galaxies that form the lensing planes.
    cosmological_model : astropy.cosmology.FLRW
        The assumed cosmology for this ray-tracing calculation.
    """
    pass


class TraceImageAndSource(object):

    def __init__(self, lens_galaxies, source_galaxies, image_plane_coordinates, border_setup=None):
        """The ray-tracing calculations, defined by a lensing system with just one image_grid-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need \
        known redshift measurements. For computational efficiency, it is recommend this ray-tracing class is used for \
        lens modeling, provided cosmological information is not necessary.

        Parameters
        ----------
        image_plane_coordinates : PlaneCoordinates
            The image_grid of the ray-tracing calculation, (includes the image_grid-grid, sub_grid-grid, sparse_grid-grid, \
            blurring_grid region etc.), which begins in the image_grid-plane.
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the image_grid-plane.
        source_galaxies : [Galaxy]
            The list of source galaxies in the source-plane.
        """
        self.image_plane = ImagePlane(lens_galaxies, image_plane_coordinates)

        source_plane_coordinates = self.trace_to_next_plane()

        self.source_plane = SourcePlane(source_galaxies, source_plane_coordinates, border_setup)

    def trace_to_next_plane(self):
        """Trace the image_grid pixel image_grid to the next plane, the source-plane."""

        coordinates = np.subtract(self.image_plane.coordinates.image_grid, self.image_plane.deflection_angles.image_grid)

        if self.image_plane.coordinates.sub_grid is not None:
            sub_coordinates = np.subtract(self.image_plane.coordinates.sub_grid, self.image_plane.deflection_angles.sub_grid)
        else:
            sub_coordinates = None

        if self.image_plane.coordinates.sparse_grid is not None:
            sparse_coordinates = np.subtract(self.image_plane.coordinates.sparse_grid, self.image_plane.deflection_angles. sparse_grid)
        else:
            sparse_coordinates = None

        if self.image_plane.coordinates.blurring_grid is not None:
            blurring_coordinates = np.subtract(self.image_plane.coordinates.blurring_grid, self.image_plane.deflection_angles. blurring_grid)
        else:
            blurring_coordinates = None

        return PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates, blurring_coordinates)

class PlaneCoordinates(geometry_profiles.Profile):

    def __init__(self, image_grid, sub_grid=None, sparse_grid=None, blurring_grid=None, centre=(0.0, 0.0)):
        """Represents the image_grid during ray-tracing.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the plane.
        image_grid : ndarray
            The x and y image_grid in the plane.
        sub_grid : ndarray
            The x and y sub_grid-image_grid in the plane.
        sparse_grid : ndarray
            The x and y sparse_grid-image_grid in the plane.
        blurring_grid : ndarray
            The x and y blurring_grid region image_grid of the plane.
        centre : (float, float)
            The centre of the plane.
        """

        super(PlaneCoordinates, self).__init__(centre)

        self.image_grid = image_grid
        self.sub_grid = sub_grid
        self.sparse_grid = sparse_grid
        self.blurring_grid = blurring_grid

    def deflection_angles_for_galaxies(self, lens_galaxies):

        deflection_angles = sum(map(lambda lens : lens.deflection_angles_array(self.image_grid), lens_galaxies))

        if self.sub_grid is not None:
            sub_deflection_angles = sum(map(lambda lens: lens.deflection_angles_sub_array(self.sub_grid), lens_galaxies))
        else:
            sub_deflection_angles = None

        if self.sparse_grid is not None:
            sparse_deflection_angles = sum(map(lambda lens: lens.deflection_angles_array(self.sparse_grid), lens_galaxies))
        else:
            sparse_deflection_angles = None

        if self.blurring_grid is not None:
            blurring_deflection_angles = sum(map(lambda lens: lens.deflection_angles_array(self.blurring_grid), lens_galaxies))
        else:
            blurring_deflection_angles = None

        return PlaneDeflectionAngles(deflection_angles, sub_deflection_angles, sparse_deflection_angles,
                                     blurring_deflection_angles)


class PlaneDeflectionAngles(object):

    def __init__(self, image_grid, sub_grid=None, sparse_grid=None, blurring_grid=None):
        """Represents the image_grid during ray-tracing.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the plane.
        image_grid : ndarray
            The x and y image_grid in the plane.
        sub_grid : ndarray
            The x and y sub_grid-image_grid in the plane.
        sparse_grid : ndarray
            The x and y sparse_grid-image_grid in the plane.
        blurring_grid : ndarray
            The x and y blurring_grid region image_grid of the plane.
        centre : (float, float)
            The centre of the plane.
        """

        self.image_grid = image_grid
        self.sub_grid = sub_grid
        self.sparse_grid = sparse_grid
        self.blurring_grid = blurring_grid


class BorderSetup(object):

    def __init__(self, border_pixels, polynomial_degree=3, centre=(0.0, 0.0)):

        self.border_pixels = border_pixels
        self.polynomial_degree = polynomial_degree
        self.centre = centre

    def from_image_grid(self, image_grid):


        coordinates = np.zeros((len(self.border_pixels), 2))

        for i, pixel in enumerate(self.border_pixels):
            coordinates[i, :] = image_grid[pixel, :]

        return PlaneBorder(coordinates, self.polynomial_degree, self.centre)


class PlaneBorder(geometry_profiles.Profile):

    def __init__(self, coordinates, polynomial_degree, centre=(0.0, 0.0)):
        """ The border of a plane, which can be used to relocate ray-traced pixels outside of the border to its edge.

        This is required to ensure highly demagnified pixels in the centre of the image_grid do not bias a source pixelization.

        Parameters
        ----------
        border_pixels : np.ndarray
            The the border source pixels, specified by their 1D index in *image_grid*.
        polynomial_degree : int
            The degree of the polynomial used to fit the source-plane border edge.
        """

        super(PlaneBorder, self).__init__(centre)

        self.thetas = list(map(lambda r: self.coordinates_angle_from_x(r), coordinates))
        self.radii = list(map(lambda r: self.coordinates_to_radius(r), coordinates))
        self.polynomial = np.polyfit(self.thetas, self.radii, polynomial_degree)

    def coordinates_angle_from_x(self, coordinates):
        """
        Compute the angle in degrees between the image_grid and plane positive x-axis, defined counter-clockwise.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid of the plane.

        Returns
        ----------
        The angle between the image_grid and the x-axis.
        """
        shifted_coordinates = self.coordinates_to_centre(coordinates)
        theta_from_x = np.degrees(np.arctan2(shifted_coordinates[1], shifted_coordinates[0]))
        if theta_from_x < 0.0:
            theta_from_x += 360.
        return theta_from_x

    def radius_at_theta(self, theta):
        """For a an angle theta from the x-axis, return the setup_border_pixels radius via the polynomial fit"""
        return np.polyval(self.polynomial, theta)

    def move_factor(self, coordinate):
        """Get the move factor of a coordinate.
         A move-factor defines how far a coordinate outside the source-plane setup_border_pixels must be moved in order to lie on it.
         PlaneCoordinates already within the setup_border_pixels return a move-factor of 1.0, signifying they are already within the \
         setup_border_pixels.

        Parameters
        ----------
        coordinate : (float, float)
            The x and y image_grid of the pixel to have its move-factor computed.
        """
        theta = self.coordinates_angle_from_x(coordinate)
        radius = self.coordinates_to_radius(coordinate)

        border_radius = self.radius_at_theta(theta)

        if radius > border_radius:
            return border_radius / radius
        else:
            return 1.0

    def relocated_coordinate(self, coordinate):
        """Get a coordinate relocated to the source-plane setup_border_pixels if initially outside of it.

        Parameters
        ----------
        coordinate : ndarray[float, float]
            The x and y image_grid of the pixel to have its move-factor computed.
        """
        move_factor = self.move_factor(coordinate)
        return coordinate[0] * move_factor, coordinate[1] * move_factor


class Plane(object):
    """Represents a plane of galaxies and coordinates.

    Parameters
    ----------
    galaxies : [Galaxy]
        The galaxies in the plane.
    plane_coordinates : PlaneCoordinates
        The x and y image_grid in the plane. Includes all image_grid e.g. the image_grid, sub_grid-grid, sparse_grid-grid, etc.
    """
    def __init__(self, galaxies, plane_coordinates, border_setup=None):

        self.galaxies = galaxies

        self.coordinates = plane_coordinates

        if border_setup is not None:
            self.border = border_setup.from_image_grid(self.coordinates.image_grid)

    def coordinates_after_border_relocation(self):

        image_grid = np.zeros(self.coordinates.image_grid.shape)

        for (i, coordinate) in enumerate(self.coordinates.image_grid):
            image_grid[i] = self.border.relocated_coordinate(coordinate)

        # TODO : Speed up using plane_image_grid to avoid scanning all sub_grid-pixels

        if self.coordinates.sub_grid is not None:
            sub_grid = np.zeros(self.coordinates.sub_grid.shape)
            for image_pixel in range(len(self.coordinates.sub_grid)):
                for (sub_pixel, sub_coordinate) in enumerate(self.coordinates.sub_grid[image_pixel]):
                    sub_grid[image_pixel, sub_pixel] = self.border.relocated_coordinate(sub_coordinate)
        else:
            sub_grid = None

        if self.coordinates.sparse_grid is not None:
            sparse_grid = np.zeros(self.coordinates.sparse_grid.shape)
            for (i, sparse_coordinate) in enumerate(self.coordinates.sparse_grid):
                sparse_grid[i] = self.border.relocated_coordinate(sparse_coordinate)
        else:
            sparse_grid = None

        return PlaneCoordinates(image_grid, sub_grid, sparse_grid, None)


class LensPlane(Plane):

    def __init__(self, galaxies, plane_coordinates, border_setup=None):
        """Represents a lens-plane, a set of galaxies and coordinates at an intermediate redshift in the lens \
        ray-tracing calculation.

        A lens-plane is not the final ray-tracing plane and its coordinates will be traced another, higher \
        redshift, plane. Thus, the deflection angles due to the plane's galaxies are calculated.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the image_grid-plane.
        plane_coordinates : PlaneCoordinates
            The x and y image_grid in the plane. Includes all image_grid e.g. the image_grid, sub_grid-grid, sparse_grid-grid, etc.
        """

        super(LensPlane, self).__init__(galaxies, plane_coordinates, border_setup)

        self.deflection_angles = self.coordinates.deflection_angles_for_galaxies(galaxies)


class ImagePlane(LensPlane):

    def __init__(self, galaxies, plane_coordinates):
        """Represents an image_grid-plane, a set of galaxies and coordinates at the lowest redshift in the lens \
        ray-tracing calculation.

        The image_grid-plane is, by definition, a lens-plane, thus the deflection angles at each coordinates are computed.

        The image_grid-plane coodinates are defined on the observed image_grid's uniform regular grid. Calculating its light \
        profiles therefore exploits this uniformity to perform more efficient and precise calculations.

        The light profiles of galaxies at higher redshifts (and therefore in different lens-planes) can be assigned to \
        the ImagePlane instead. This occurs when:

        1) The efficiency and precision offered by computing the light profile on a uniform grid is preferred and \
        won't lead noticeable inaccuracy. For example, computing the light profile of main lens galaxy, ignoring \
        minor lensing effects due to a low mass foreground substructure.

        2) If evaluating the light profile in its lens-plane is inaccurate. For example, when modeling the \
        point-source images of a lensed quasar, effects like micro-lensing means lens-plane modeling will be inaccurate.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the image_grid-plane.
        plane_coordinates : PlaneCoordinates
            The x and y image_grid in the plane. Includes all image_grid e.g. the image_grid, sub_grid-grid, sparse_grid-grid, etc.
        """

        super(ImagePlane, self).__init__(galaxies, plane_coordinates, None)


class SourcePlane(Plane):

    def __init__(self, galaxies, plane_coordinates, border_setup=None):
        """Represents a source-plane, a set of galaxies and coordinates at the highest redshift in the lens \
        ray-tracing calculation.

        A source-plane is the final ray-tracing plane, thus the deflection angles due to the plane's galaxies are \
        not calculated.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the source-plane.
        plane_coordinates : PlaneCoordinates
            The x and y image_grid in the plane. Includes all image_grid e.g. the image_grid, sub_grid-grid, sparse_grid-grid, etc.
        """
        super(SourcePlane, self).__init__(galaxies, plane_coordinates, border_setup)