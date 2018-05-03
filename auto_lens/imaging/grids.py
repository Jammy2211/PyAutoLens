import numpy as np

from auto_lens.imaging import imaging
from auto_lens.profiles import geometry_profiles


class GridCoordsCollection(object):

    def __init__(self, image, sub=None, blurring=None):
        """A collection of grids which contain the coordinates of the image_coords, image_coords sub-grid_coords, blurring regions etc. \
         These grids are all passed through the ray-tracing module to set up the image_coords, lens and source planes.
         
        Parameters
        -----------
        image : GridCoordsImage
            A grid_coords containing the image_coords coordinates.
        sub : GridCoordsImageSub
            A grid_coords containing the image_coords sub-coordinates.
        blurring : GridCoordsBlurring
            A grid_coords containing the blurring region coordinates.
        """

        self.image = image
        self.sub = sub
        self.blurring = blurring

    @classmethod
    def from_mask(cls, mask, sub_grid_size=None, blurring_size=None):
        """Setup the collection of grids using the image mask.

        Parameters
        -----------
        image : GridCoordsImage
            A grid_coords containing the image coordinates.
        sub : GridCoordsImageSub
            A grid_coords containing the image sub-coordinates.
        blurring : GridCoordsBlurring
            A grid_coords containing the blurring region coordinates.
        """

        image = GridCoordsImage.from_mask(mask)

        if sub_grid_size is None:
            sub = None
        elif sub_grid_size is not None:
            sub = GridCoordsImageSub.from_mask(mask, sub_grid_size)

        if blurring_size is None:
            blurring = None
        elif blurring_size is not None:
            blurring = GridCoordsBlurring.from_mask(mask, blurring_size)

        return GridCoordsCollection(image, sub, blurring)

    def setup_all_deflections_grids(self, galaxies):
        """Compute the deflection angles of every grids (by integrating the mass profiles of the input galaxies) \
        and set these up as a new collection of grids."""

        image = self.image.setup_deflections_grid(galaxies)

        if self.sub is None:
            sub = None
        elif self.sub is not None:
            sub = self.sub.setup_deflections_grid(galaxies)

        if self.blurring is None:
            blurring = None
        elif self.blurring is not None:
            blurring = self.blurring.setup_deflections_grid(galaxies)

        return GridCoordsCollection(image, sub, blurring)

    def setup_all_traced_grids(self, deflections):
        """Setup a new collection of grids by tracing their coordinates using a set of deflection angles."""
        image = self.image.setup_traced_grid(deflections.image)

        if self.sub is None:
            sub = None
        elif self.sub is not None:
            sub = self.sub.setup_traced_grid(deflections.sub)

        if self.blurring is None:
            blurring = None
        elif self.blurring is not None:
            blurring = self.blurring.setup_traced_grid(deflections.blurring)

        return GridCoordsCollection(image, sub, blurring)


class GridCoords(object):

    def __init__(self, grid_coords):
        """Abstract base class for a set of grid coordinates, which store the arc-second coordinates of different \
        regions of an image. These are the coordinates where ray-tracing and lensing analysis are performed.

        The different regions each set of grid coordinates correspond to controll different aspects of the \
        analysis. For example, the image-sub coordinates are used to compute images on a sub-gridded uniform grid and \
        the blurring coordinates include areas outside the image mask but close enough their light is blurred in by \
        the PSF.

        Each grid is stored as a structured array of coordinates, chosen for efficient ray-tracing \
        calculations. Coordinates are defined from the top-left corner, such that pixels in the top-left corner of an \
        image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds.
        """
        self.grid_coords = grid_coords


class GridCoordsRegular(GridCoords):

    def __init__(self, grid_coords):
        """Abstract class for a regular grid_coords, where pixel coordinates are represented by just one coordinate at \
        the centre of its respective pixel.

        A regular grid_coords is a NumPy array of dimensions [image_pixels, 2]. Therefore, the first element maps to the \
        image pixel index, and second element to its (x,y) arc second coordinates. For example, the value [3,1] gives \
        the 4th image pixel's y coordinate.

        Parameters
        -----------
        grid_coords : np.ndarray
            The regular grid_coords coordinates.
        """

        super(GridCoordsRegular, self).__init__(grid_coords)

    def intensities_via_grid(self, galaxies):
        """Compute the intensity in each pixel using the coordinates of the regular grid_coords"""
        return sum(map(lambda galaxy : self.evaluate_func_on_grid(func=galaxy.intensity_at_coordinates,
                                                                  output_shape=self.grid_coords.shape[0]), galaxies))

    def deflections_on_grid(self, galaxies):
        """Compute the deflection angles of each coordinate of the regular grid_coords"""
        return sum(map(lambda galaxy : self.evaluate_func_on_grid(func=galaxy.deflections_at_coordinates,
                                                                  output_shape=self.grid_coords.shape), galaxies))

    def evaluate_func_on_grid(self, func, output_shape):
        """Compute a set of values (e.g. intensities or deflections angles) for a light or mass profile, at the set of \
        coordinates defined by the regular grid_coords.
        """
        grid_values = np.zeros(output_shape)

        for pixel_no, coordinate in enumerate(self.grid_coords):
            grid_values[pixel_no] = func(coordinates=coordinate)

        return grid_values


class GridCoordsSub(GridCoords):

    def __init__(self, grid_coords, grid_sub_size):
        """Abstract class for a sub grid_coords, where pixel coordinates are represented by a uniform grid_coords of coordinates \
        within the pixel.

        A regular grid_coords is a NumPy array of dimensions [image_pixels, 2]. Therefore, the first element maps to the \
        image pixel index, and second element to its central (x,y) arc second coordinates. For example, the value [3,1]
        gives the 4th image pixel's y coordinate.

        A sub grid_coords is a NumPy array of dimensions [image_pixels, sub_grid_pixels, 2]. Therefore, the first element \
        maps to the image pixel index, the second element to the sub-pixel index and third element to that sub pixel's \
        (x,y) arc second coordinates. For example, the value [3, 6, 1] gives the 4th image pixel's 7th sub-pixel's \
        y coordinate.

        Parameters
        -----------
        grid_coords : np.ndarray
            The sub-grid_coords coordinates.
        grid_sub_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid_coords of each image pixel.
        """
        self.sub_grid_size = grid_sub_size
        self.sub_grid_size_squared = grid_sub_size ** 2.0
        super(GridCoordsSub, self).__init__(grid_coords)

    def intensities_via_grid(self, galaxies):
        """Compute the intensity in each pixel using the coordinates of the sub grid_coords. This routine takes the mean \
         value of the intensities at sub grid_coords coordinates to compute the final intensity."""

        sub_intensitites = sum(map(lambda galaxy : self.evaluate_func_on_grid(func=galaxy.intensity_at_coordinates,
                                                                              output_shape=self.grid_coords.shape[0:2]), galaxies))

        intensitites = np.zeros(self.grid_coords.shape[0])

        for pixel_no, intensities_sub_pixel in enumerate(sub_intensitites):
            intensitites[pixel_no] = np.sum(intensities_sub_pixel) / self.sub_grid_size_squared

        return intensitites

    def deflections_on_grid(self, galaxies):
        """Compute the deflection angles of each coordinate of the sub grid_coords"""
        return sum(map(lambda galaxy : self.evaluate_func_on_grid(func=galaxy.deflections_at_coordinates,
                                                                  output_shape=self.grid_coords.shape), galaxies))

    def evaluate_func_on_grid(self, func, output_shape):
        """Compute a set of values (e.g. intensities or deflections angles) for a light or mass profile, at the set of \
        coordinates defined by a sub-grid_coords.
        """

        sub_grid_values = np.zeros(output_shape)

        for pixel_no, pixel_sub_grid in enumerate(self.grid_coords):
            for sub_pixel_no, sub_coordinate in enumerate(pixel_sub_grid):
                sub_grid_values[pixel_no, sub_pixel_no] = func(coordinates=sub_coordinate)

        return sub_grid_values


class GridCoordsImage(GridCoordsRegular):

    def __init__(self, grid_coords):
        """The image grid_coords, representing all pixel coordinates in an image.

        Parameters
        -----------
        grid_coords : np.ndarray
            The regular grid_coords of image coordinates.
        """

        super(GridCoordsImage, self).__init__(grid_coords)

    @classmethod
    def from_mask(cls, mask):
        """ Given an image.Mask, setup the image grid_coords using the center of every unmasked pixel.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the image grid_coords is computed for and the image's data grid_coords.
        """
        return GridCoordsImage(mask.compute_grid_coords_image())

    def setup_deflections_grid(self, galaxies):
        """Setup a new image grid_coords of deflection angle coordinates, by integrating the mass profiles of a set of \
        galaxies."""
        return GridCoordsImage(self.deflections_on_grid(galaxies))

    def setup_traced_grid(self, grid_deflections):
        """Setup a new image grid_coords of coordinates, by tracing its coordinates by a set of deflection angles."""
        return GridCoordsImage(np.subtract(self.grid_coords, grid_deflections.grid_coords))


class GridCoordsImageSub(GridCoordsSub):

    def __init__(self, grid_coords, grid_sub_size):
        """The image sub-grid_coords, representing all sub-pixel coordinates in an image.

        Parameters
        -----------
        grid_coords : np.ndarray
            The sub-grid_coords of image coordinates.
        grid_sub_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid_coords of each image pixel.
        """
        super(GridCoordsImageSub, self).__init__(grid_coords, grid_sub_size)

    @classmethod
    def from_mask(cls, mask, sub_grid_size):
        """ Given an image.Mask, compute the image sub-grid_coords using the center of every unmasked pixel and an input \
        sub-grid_coords size.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the image sub-grid_coords is computed for and the image's data grid_coords.
        sub_grid_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid_coords of each image pixel.
        """
        return GridCoordsImageSub(mask.compute_grid_coords_image_sub(sub_grid_size), sub_grid_size)

    def setup_deflections_grid(self, galaxies):
        """Setup a new sub grid_coords of deflection angle coordinates, by integrating the mass profiles of a set of \
        galaxies."""
        return GridCoordsImageSub(self.deflections_on_grid(galaxies), self.sub_grid_size)

    def setup_traced_grid(self, deflection_grid):
        """Setup a new image grid_coords of sub-coordinates, by tracing its sub-coordinates by a set of deflection angles."""
        return GridCoordsImageSub(np.subtract(self.grid_coords, deflection_grid.grid_coords), self.sub_grid_size)


class GridCoordsBlurring(GridCoordsRegular):

    def __init__(self, grid_coords):
        """The blurring grid_coords, representing all pixel coordinates in the regions of an image which are outside of the \
         mask but will have a fraction of their light blurred into the mask via PSF convolution.

        Parameters
        -----------
        grid_coords : np.ndarray
            The regular grid_coords of blurring pixel coordinates.
        """

        super(GridCoordsBlurring, self).__init__(grid_coords)

    @classmethod
    def from_mask(cls, mask, psf_size):
        """ Given an image.Mask, compute the blurring analysis grid_coords by locating all pixels which are within the \
        psf size of the mask.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the blurring grid_coords is computed for and the image's data grid_coords.
        psf_size : (int, int)
           The size of the psf which defines the blurring region (e.g. the pixel_dimensions of the PSF)
        """
        if psf_size[0] % 2 == 0 or psf_size[1] % 2 == 0:
            raise imaging.MaskException("psf_size of exterior region must be odd")

        return GridCoordsBlurring(mask.compute_grid_coords_blurring(psf_size))

    def setup_deflections_grid(self, galaxies):
        """Setup a new blurring grid_coords of deflection angle coordinates, by integrating the mass profiles of a set of \
        galaxies."""
        return GridCoordsBlurring(self.deflections_on_grid(galaxies))

    def setup_traced_grid(self, deflection_grid):
        """Setup a new blurring grid_coords of coordinates, by tracing its coordinates by a set of deflecton angles."""
        return GridCoordsBlurring(np.subtract(self.grid_coords, deflection_grid.grid_coords))


class GridData(object):

    def __init__(self, grid_data):
        """Abstract base class for the grid of a data-set (e.g. the image, noise, exposure times).

        Each grid is stored as a 1D array of the data values, which ensures efficient calculations during lens \
        analysis.
        """
        self.grid_data = grid_data

    @classmethod
    def from_mask(cls, data, mask):
        """ Given an image.Mask, setup the data grid using the every unmasked pixel.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the image grid_coords is computed for and the image's data grid_coords.
        """
        return GridData(mask.compute_grid_data(data))

class GridMapperSparse(object):
    """GridCoords mappings between the sparse grid_coords and image grid_coords."""

    def __init__(self, sparse_to_image, image_to_sparse):

        self.sparse_to_image = sparse_to_image
        self.image_to_sparse = image_to_sparse

    @classmethod
    def from_mask(cls, mask, sparse_grid_size):
        """ Given an image.Mask, compute the sparse mapper of the image by inputting the sparse grid_coords size and finding \
        all image pixels which are on the sparse grid_coords.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the blurring grid_coords is computed for and the image's data grid_coords.
        """
        sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(sparse_grid_size)
        return GridMapperSparse(sparse_to_image, image_to_sparse)


class GridBorder(geometry_profiles.Profile):

    # TODO : Could speed this up by only looking and relocating image pixels within a certain radius of the image
    # TODO : centre. This would add a central_pixels lists to the input.

    def __init__(self, border_pixels, polynomial_degree=3, centre=(0.0, 0.0)):
        """ The border of a grid_coords, which can be used to relocate coordinates outside of the border to its edge.

        This is required to ensure highly demagnified pixels in the centre of the image_grid do not bias a source \
        pixelization.

        Parameters
        ----------
        border_pixels : np.ndarray
            The the border source pixels, specified by their 1D index in *image_grid*.
        polynomial_degree : int
            The degree of the polynomial used to fit the source-plane border edge.
        """

        super(GridBorder, self).__init__(centre)

        self.border_pixels = border_pixels
        self.polynomial_degree = polynomial_degree
        self.centre = centre

    def relocate_coordinates_outside_border(self, coordinates):
        """For an input coordinates, return a coordinates where all coordinates outside the border are relocated to its edge."""

        self.polynomial_fit_to_border(coordinates)

        relocated_coordinates = np.zeros(coordinates.shape)

        for (i, coordinate) in enumerate(coordinates):
            relocated_coordinates[i] = self.relocated_coordinate(coordinate)

        return relocated_coordinates

    def relocate_sub_coordinates_outside_border(self, coordinates, sub_coordinates):
        """For an input sub-coordinates, return a coordinates where all sub-coordinates outside the border are relocated to its edge.
        """

        # TODO : integrate these as functions into GridCoords and SubGrid, or pass in a GridCoords / SubGrid?

        self.polynomial_fit_to_border(coordinates)

        relocated_sub_coordinates = np.zeros(sub_coordinates.shape)

        for image_pixel in range(len(coordinates)):
            for (sub_pixel, sub_coordinate) in enumerate(sub_coordinates[image_pixel]):
                relocated_sub_coordinates[image_pixel, sub_pixel] = self.relocated_coordinate(sub_coordinate)

        return relocated_sub_coordinates

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

    def polynomial_fit_to_border(self, coordinates):

        border_coordinates = coordinates[self.border_pixels]

        self.thetas = list(map(lambda r: self.coordinates_angle_from_x(r), border_coordinates))
        self.radii = list(map(lambda r: self.coordinates_to_radius(r), border_coordinates))
        self.polynomial = np.polyfit(self.thetas, self.radii, self.polynomial_degree)

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


class GridException(Exception):
    pass