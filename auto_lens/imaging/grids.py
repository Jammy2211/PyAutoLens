import numpy as np

from auto_lens.imaging import imaging
from auto_lens.profiles import geometry_profiles


class GridCollection(object):

    def __init__(self, image, sub=None, blurring=None):
        """A collection of grids which contain the coordinates of the image, image sub-grid, blurring regions etc. \
         These grids are all passed through the ray-tracing module to set up the image, lens and source planes.
         
        Parameters
        -----------
        image : GridImage
            A grid containing the image coordinates.
        sub : GridImageSub
            A grid containing the image sub-coordinates.
        blurring : GridBlurring
            A grid containing the blurring region coordinates.
        """

        self.image = image
        self.sub = sub
        self.blurring = blurring

    @classmethod
    def from_mask(cls, mask, sub_grid_size=None, blurring_size=None):
        """Setup the collection of grids using the image mask.

        Parameters
        -----------
        image : GridImage
            A grid containing the image coordinates.
        sub : GridImageSub
            A grid containing the image sub-coordinates.
        blurring : GridBlurring
            A grid containing the blurring region coordinates.
        """

        image = GridImage.from_mask(mask)

        if sub_grid_size is None:
            sub = None
        elif sub_grid_size is not None:
            sub = GridImageSub.from_mask(mask, sub_grid_size)

        if blurring_size is None:
            blurring = None
        elif blurring_size is not None:
            blurring = GridBlurring.from_mask(mask, blurring_size)

        return GridCollection(image, sub, blurring)

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

        return GridCollection(image, sub, blurring)

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

        return GridCollection(image, sub, blurring)


class Grid(object):

    def __init__(self, grid):
        """Abstract base class for a grid, which store the pixel coordinates of different regions of an \
        image and where the ray-tracing and lensing analysis are performed.

        The different regions represented by each grid are used for controlling different aspects of the \
        analysis (e.g. the image, the image sub-grid, the clustering grid, etc.)

        The grids are stored as a structured array of coordinates, chosen for efficient ray-tracing \
        calculations. Coordinates are defined from the top-left corner, such that pixels in the top-left corner of an \
        image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds.
        """
        self.grid = grid


class GridRegular(Grid):

    def __init__(self, grid):
        """Abstract class for a regular grid, where pixel coordinates are represented by just one coordinate at \
        the centre of its respective pixel.

        A regular grid is a NumPy array of dimensions [image_pixels, 2]. Therefore, the first element maps to the \
        image pixel index, and second element to its (x,y) arc second coordinates. For example, the value [3,1] gives \
        the 4th image pixel's y coordinate.

        Parameters
        -----------
        grid : np.ndarray
            The regular grid coordinates.
        """

        super(GridRegular, self).__init__(grid)

    def deflections_on_grid(self, galaxies):
        return sum(map(lambda galaxy : self.evaluate_func_on_grid(galaxy.deflections_at_coordinates), galaxies))

    def evaluate_func_on_grid(self, func):
        """Compute a set of values (e.g. intensities or deflections angles) for a light or mass profile, at the set of \
        coordinates defined by the regular grid.
        """
        grid_values = np.zeros(self.grid.shape)

        for pixel_no, coordinate in enumerate(self.grid):
            grid_values[pixel_no] = func(coordinates=coordinate)

        return grid_values


class GridSub(Grid):

    def __init__(self, grid, sub_grid_size):
        """Abstract class for a sub grid, where pixel coordinates are represented by a uniform grid of coordinates \
        within the pixel.

        A regular grid is a NumPy array of dimensions [image_pixels, 2]. Therefore, the first element maps to the \
        image pixel index, and second element to its central (x,y) arc second coordinates. For example, the value [3,1]
        gives the 4th image pixel's y coordinate.

        A sub grid is a NumPy array of dimensions [image_pixels, sub_grid_pixels, 2]. Therefore, the first element \
        maps to the image pixel index, the second element to the sub-pixel index and third element to that sub pixel's \
        (x,y) arc second coordinates. For example, the value [3, 6, 1] gives the 4th image pixel's 7th sub-pixel's \
        y coordinate.

        Parameters
        -----------
        grid : np.ndarray
            The sub-grid coordinates.
        sub_grid_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid of each image pixel.
        """
        self.sub_grid_size = sub_grid_size
        super(GridSub, self).__init__(grid)

    def deflections_on_grid(self, galaxies):
        return sum(map(lambda galaxy : self.evaluate_func_on_grid(galaxy.deflections_at_coordinates), galaxies))

    def evaluate_func_on_grid(self, func):
        """Compute a set of values (e.g. intensities or deflections angles) for a light or mass profile, at the set of \
        coordinates defined by a sub-grid.
        """
        grid_values = np.zeros(self.grid.shape)

        for pixel_no, pixel_sub_grid in enumerate(self.grid):
            for sub_pixel_no, sub_coordinate in enumerate(pixel_sub_grid):
                grid_values[pixel_no, sub_pixel_no] = func(coordinates=sub_coordinate)

        return grid_values


class GridImage(GridRegular):

    def __init__(self, grid):
        """The image grid, representing all pixel coordinates in an image.

        Parameters
        -----------
        grid : np.ndarray
            The regular grid of image coordinates.
        """

        super(GridImage, self).__init__(grid)

    @classmethod
    def from_mask(cls, mask):
        """ Given an image.Mask, setup the image grid using the center of every unmasked pixel.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the image grid is computed for and the image's data grid.
        """
        return GridImage(mask.compute_image_grid())

    def setup_deflections_grid(self, galaxies):
        """Setup a new image grid of deflection angle coordinates, by integrating the mass profiles of a set of \
        galaxies."""
        return GridImage(self.deflections_on_grid(galaxies))

    def setup_traced_grid(self, grid_deflections):
        """Setup a new image grid of coordinates, by tracing its coordinates by a set of deflection angles."""
        return GridImage(np.subtract(self.grid, grid_deflections.grid))


class GridImageSub(GridSub):

    def __init__(self, grid, sub_grid_size):
        """The image sub-grid, representing all sub-pixel coordinates in an image.

        Parameters
        -----------
        grid : np.ndarray
            The sub-grid of image coordinates.
        sub_grid_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid of each image pixel.
        """
        super(GridImageSub, self).__init__(grid, sub_grid_size)

    @classmethod
    def from_mask(cls, mask, sub_grid_size):
        """ Given an image.Mask, compute the image sub-grid using the center of every unmasked pixel and an input \
        sub-grid size.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the image sub-grid is computed for and the image's data grid.
        sub_grid_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid of each image pixel.
        """
        return GridImageSub(mask.compute_image_sub_grid(sub_grid_size), sub_grid_size)

    def setup_deflections_grid(self, galaxies):
        """Setup a new sub grid of deflection angle coordinates, by integrating the mass profiles of a set of \
        galaxies."""
        return GridImageSub(self.deflections_on_grid(galaxies), self.sub_grid_size)

    def setup_traced_grid(self, deflection_grid):
        """Setup a new image grid of sub-coordinates, by tracing its sub-coordinates by a set of deflection angles."""
        return GridImageSub(np.subtract(self.grid, deflection_grid.grid), self.sub_grid_size)


class GridBlurring(GridRegular):

    def __init__(self, grid):
        """The blurring grid, representing all pixel coordinates in the regions of an image which are outside of the \
         mask but will have a fraction of their light blurred into the mask via PSF convolution.

        Parameters
        -----------
        grid : np.ndarray
            The regular grid of blurring pixel coordinates.
        """

        super(GridBlurring, self).__init__(grid)

    @classmethod
    def from_mask(cls, mask, psf_size):
        """ Given an image.Mask, compute the blurring analysis grid by locating all pixels which are within the \
        psf size of the mask.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the blurring grid is computed for and the image's data grid.
        psf_size : (int, int)
           The size of the psf which defines the blurring region (e.g. the pixel_dimensions of the PSF)
        """
        if psf_size[0] % 2 == 0 or psf_size[1] % 2 == 0:
            raise imaging.MaskException("psf_size of exterior region must be odd")

        return GridBlurring(mask.compute_blurring_grid(psf_size))

    def setup_deflections_grid(self, galaxies):
        """Setup a new blurring grid of deflection angle coordinates, by integrating the mass profiles of a set of \
        galaxies."""
        return GridBlurring(self.deflections_on_grid(galaxies))

    def setup_traced_grid(self, deflection_grid):
        """Setup a new blurring grid of coordinates, by tracing its coordinates by a set of deflecton angles."""
        return GridBlurring(np.subtract(self.grid, deflection_grid.grid))


class GridMapperSparse(object):
    """Grid mappings between the sparse grid and image grid."""

    def __init__(self, sparse_to_image, image_to_sparse):

        self.sparse_to_image = sparse_to_image
        self.image_to_sparse = image_to_sparse

    @classmethod
    def from_mask(cls, mask, sparse_grid_size):
        """ Given an image.Mask, compute the sparse mapper of the image by inputting the sparse grid size and finding \
        all image pixels which are on the sparse grid.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the blurring grid is computed for and the image's data grid.
        """
        sparse_to_image, image_to_sparse = mask.compute_sparse_mappers(sparse_grid_size)
        return GridMapperSparse(sparse_to_image, image_to_sparse)


class GridBorder(geometry_profiles.Profile):

    # TODO : Could speed this up by only looking and relocating image pixels within a certain radius of the image
    # TODO : centre. This would add a central_pixels lists to the input.

    def __init__(self, border_pixels, polynomial_degree=3, centre=(0.0, 0.0)):
        """ The border of a grid, which can be used to relocate coordinates outside of the border to its edge.

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

        # TODO : integrate these as functions into Grid and SubGrid, or pass in a Grid / SubGrid?

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