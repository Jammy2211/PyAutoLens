from auto_lens.profiles import geometry_profiles

import numpy as np

class RayTrace(object):
    """The ray-tracing calculations, defined by the image and source planes of every galaxy in this lensing system.
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

    def __init__(self, lens_galaxies, source_galaxies, image_plane_coordinates):
        """The ray-tracing calculations, defined by a lensing system with just one image-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need \
        known redshift measurements. For computational efficiency, it is recommend this ray-tracing class is used for \
        lens modeling, provided cosmological information is not necessary.

        Parameters
        ----------
        image_plane_coordinates : PlaneCoordinates
            The image of the ray-tracing calculation, (includes the image-grid, sub-grid, sparse-grid, \
            blurring region etc.), which begins in the image-plane.
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the image-plane.
        source_galaxies : [Galaxy]
            The list of source galaxies in the source-plane.
        """
        self.image_plane = ImagePlane(lens_galaxies, image_plane_coordinates)

        source_plane_coordinates = self.trace_to_next_plane()

        self.source_plane = SourcePlane(source_galaxies, source_plane_coordinates)

    def trace_to_next_plane(self):
        """Trace the image pixel image to the next plane, the source-plane."""

        coordinates = np.subtract(self.image_plane.coordinates.image, self.image_plane.deflection_angles.image)

        if self.image_plane.coordinates.sub is not None:
            sub_coordinates = np.subtract(self.image_plane.coordinates.sub, self.image_plane.deflection_angles.sub)
        else:
            sub_coordinates = None

        if self.image_plane.coordinates.sparse is not None:
            sparse_coordinates = np.subtract(self.image_plane.coordinates.sparse, self.image_plane.deflection_angles. sparse)
        else:
            sparse_coordinates = None

        return PlaneCoordinates(coordinates, sub_coordinates, sparse_coordinates)


class PlaneCoordinates(geometry_profiles.Profile):

    def __init__(self, coordinates, sub_coordinates=None, sparse_coordinates=None, blurring_coordinates=None,
                 centre=(0.0, 0.0)):
        """Represents the image during ray-tracing.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the plane.
        coordinates : ndarray
            The x and y image in the plane.
        sub_coordinates : ndarray
            The x and y sub-image in the plane.
        sparse_coordinates : ndarray
            The x and y sparse-image in the plane.
        blurring_coordinates : ndarray
            The x and y blurring region image of the plane.
        centre : (float, float)
            The centre of the plane.
        """

        super(PlaneCoordinates, self).__init__(centre)

        self.image = coordinates
        self.sub = sub_coordinates
        self.sparse = sparse_coordinates
        self.blurring = blurring_coordinates

    def deflection_angles_for_galaxies(self, lens_galaxies):

        deflection_angles = sum(map(lambda lens : lens.deflection_angles_array(self.image), lens_galaxies))

        if self.sub is not None:
            sub_deflection_angles = sum(map(lambda lens: lens.deflection_angles_sub_array(self.sub), lens_galaxies))
        else:
            sub_deflection_angles = None

        if self.sparse is not None:
            sparse_deflection_angles = sum(map(lambda lens: lens.deflection_angles_array(self.sparse), lens_galaxies))
        else:
            sparse_deflection_angles = None

        if self.blurring is not None:
            blurring_deflection_angles = sum(map(lambda lens: lens.deflection_angles_array(self.blurring), lens_galaxies))
        else:
            blurring_deflection_angles = None

        return PlaneDeflectionAngles(deflection_angles, sub_deflection_angles, sparse_deflection_angles,
                                     blurring_deflection_angles)

    def coordinates_angle_from_x(self, coordinates):
        """
        Compute the angle in degrees between the image and plane positive x-axis, defined counter-clockwise.

        Parameters
        ----------
        coordinates : ndarray
            The x and y image of the plane.

        Returns
        ----------
        The angle between the image and the x-axis.
        """
        shifted_coordinates = self.coordinates_to_centre(coordinates)
        theta_from_x = np.degrees(np.arctan2(shifted_coordinates[1], shifted_coordinates[0]))
        if theta_from_x < 0.0:
            theta_from_x += 360.
        return theta_from_x


class PlaneDeflectionAngles(object):

    def __init__(self, deflection_angles, sub_deflection_angles=None, sparse_deflection_angles=None,
                 blurring_deflection_angles=None):
        """Represents the image during ray-tracing.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the plane.
        deflection_angles : ndarray
            The x and y image in the plane.
        sub_deflection_angles : ndarray
            The x and y sub-image in the plane.
        sparse_deflection_angles : ndarray
            The x and y sparse-image in the plane.
        blurring_deflection_angles : ndarray
            The x and y blurring region image of the plane.
        centre : (float, float)
            The centre of the plane.
        """

        self.image = deflection_angles
        self.sub = sub_deflection_angles
        self.sparse = sparse_deflection_angles
        self.blurring = blurring_deflection_angles


class Plane(object):
    """Represents a plane of galaxies and its corresponding image.

    Parameters
    ----------
    galaxies : [Galaxy]
        The galaxies in the plane.
    plane_coordinates : PlaneCoordinates
        The x and y image in the plane. Includes all image e.g. the image, sub-grid, sparse-grid, etc.
    centre : (float, float)
        The centre of the plane.
    """
    def __init__(self, galaxies, plane_coordinates):

        self.galaxies = galaxies

        self.coordinates = plane_coordinates


class ImagePlane(Plane):

    def __init__(self, galaxies, plane_coordinates):
        """Represents the image-plane and its corresponding image image.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the image-plane.
        plane_coordinates : PlaneCoordinates
            The x and y image in the plane. Includes all image e.g. the image, sub-grid, sparse-grid, etc.
        centre : (float, float)
            The centre of the image-plane.
        """

        super(ImagePlane, self).__init__(galaxies, plane_coordinates)

        self.deflection_angles = self.coordinates.deflection_angles_for_galaxies(galaxies)


class SourcePlane(Plane):

    def __init__(self, galaxies, plane_coordinates):
        """
        Represents the source-plane and its corresponding traced image image.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the source-plane.
        plane_coordinates : PlaneCoordinates
            The x and y image in the plane. Includes all image e.g. the image, sub-grid, sparse-grid, etc.
        centre : (float, float)
            The centre of the source-plane.
        """
        super(SourcePlane, self).__init__(galaxies, plane_coordinates)


class SourcePlaneWithBorder(SourcePlane):

    def __init__(self, galaxies, border_pixels, polynomial_degree, plane_coordinates):
        """
        Represents the source-plane and its corresponding traced image image. \

        This source-plane has a border, such that image that trace outside of this border are relocated to the \
        border-edge, which is required to ensure highly demagnified central pixels do not bias a pixelized pixelization.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the source-plane.
        border_pixels : np.ndarray
            The the border source pixels, specified by their 1D index in *image*.
        polynomial_degree : int
            The degree of the polynomial used to fit the source-plane border edge.
        plane_coordinates : PlaneCoordinates
            The x and y image in the plane. Includes all image e.g. the image, sub-grid, sparse-grid, etc.
        centre : (float, float)
            The centre of the source-plane.
        """

        super(SourcePlaneWithBorder, self).__init__(galaxies, plane_coordinates)

        self.border_coordinates = np.zeros((len(border_pixels), 2))

        for i, border_pixel in enumerate(border_pixels):
            self.border_coordinates[i] = self.coordinates.image[border_pixel]

        self.border_thetas = list(map(lambda r: self.coordinates.coordinates_angle_from_x(r), self.border_coordinates))
        self.border_radii = list(map(lambda r: self.coordinates.coordinates_to_radius(r), self.border_coordinates))
        self.border_polynomial = np.polyfit(self.border_thetas, self.border_radii, polynomial_degree)

        for (i, coordinate) in enumerate(self.coordinates.image):
            self.coordinates.image[i] = self.relocated_coordinate(coordinate)

        # TODO : Speed up using self.image to avoid scanning all sub-pixels

        if self.coordinates.sub is not None:
            for image_pixel in range(len(self.coordinates.sub)):
                for (sub_pixel, sub_coordinate) in enumerate(self.coordinates.sub[image_pixel]):
                    self.coordinates.sub[image_pixel, sub_pixel] = self.relocated_coordinate(sub_coordinate)

        if self.coordinates.sparse is not None:
            for (i, sparse_coordinate) in enumerate(self.coordinates.sparse):
                self.coordinates.sparse[i] = self.relocated_coordinate(sparse_coordinate)

    def border_radius_at_theta(self, theta):
        """For a an angle theta from the x-axis, return the setup_border_pixels radius via the polynomial fit"""
        return np.polyval(self.border_polynomial, theta)

    def move_factor(self, coordinate):
        """Get the move factor of a coordinate.
         A move-factor defines how far a coordinate outside the source-plane setup_border_pixels must be moved in order to lie on it.
         PlaneCoordinates already within the setup_border_pixels return a move-factor of 1.0, signifying they are already within the \
         setup_border_pixels.

        Parameters
        ----------
        coordinate : (float, float)
            The x and y image of the pixel to have its move-factor computed.
        """
        theta = self.coordinates.coordinates_angle_from_x(coordinate)
        radius = self.coordinates.coordinates_to_radius(coordinate)

        border_radius = self.border_radius_at_theta(theta)

        if radius > border_radius:
            return border_radius / radius
        else:
            return 1.0

    def relocated_coordinate(self, coordinate):
        """Get a coordinate relocated to the source-plane setup_border_pixels if initially outside of it.

        Parameters
        ----------
        coordinate : ndarray[float, float]
            The x and y image of the pixel to have its move-factor computed.
        """
        move_factor = self.move_factor(coordinate)
        return coordinate[0] * move_factor, coordinate[1] * move_factor