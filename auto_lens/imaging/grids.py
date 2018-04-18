import numpy as np

from auto_lens.imaging import imaging
from auto_lens.profiles import geometry_profiles


class RayTracingGrids(object):

    def __init__(self, image, sub=None, blurring=None):

        self.image = image
        self.sub = sub
        self.blurring = blurring

    @classmethod
    def from_mask(cls, mask, sub_grid_size=None, blurring_size=None):

        image = GridImage.from_mask(mask)

        if sub_grid_size is None:
            sub = None
        elif sub_grid_size is not None:
            sub = GridImageSub.from_mask(mask, sub_grid_size)

        if blurring_size is None:
            blurring = None
        elif blurring_size is not None:
            blurring = GridBlurring.from_mask(mask, blurring_size)

        return RayTracingGrids(image, sub, blurring)


class Grid(object):

    def __init__(self, grid):
        """Abstract base class for analysis grids, which store the pixel grids of different regions of an \
        image and where the ray-tracing and lensing analysis are performed.

        The different regions represented by each analysis grid are used for controlling different aspects of the \
        analysis (e.g. the image, the image sub-grid, the clustering grid, etc.)

        The grids are stored as a structured array of pixel grids, chosen for efficient ray-tracing \
        calculations. Coordinates are defined from the top-left corner, such that pixels in the top-left corner of an \
        image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds.
        """

        self.grid = grid


class GridImage(Grid):

    def __init__(self, grid):
        """The image analysis grid, representing all pixel grids in an image where the ray-tracing and lensing
        analysis is performed.

        Parameters
        -----------
        grid : np.ndarray[image_pixels, 2]
            Array containing the image grid grids. The first elements maps to an image pixel, and second to its \
            (x,y) arc second grids. E.g. the value grid[3,1] give the 4th image pixel's y coordinate.
        """

        super(GridImage, self).__init__(grid)

    @classmethod
    def from_mask(cls, mask):
        """ Given an image.Mask, setup the image analysis grid using the center of every unmasked pixel.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the image grid is computed for and the image's data grid.
        """
        return GridImage(mask.compute_image_grid())


class GridImageSub(Grid):

    def __init__(self, grid, sub_grid_size):
        """The image analysis sub-grid, representing all pixel sub-grids in an image where the ray-tracing and \
        lensing analysis is performed.

        Parameters
        -----------
        grid : np.ndarray[image_pixels, sub_grid_size**2, 2]
            Array containing the sub-grid grids. The first elements maps to an image pixel, the second to its \
            sub-pixel and third to its (x,y) arc second grids. E.g. the value grid[3,6,1] give the 4th image \
            pixel's 7th sub-pixel's y coordinate.
        sub_grid_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid of each image pixel.
        """
        self.sub_grid_size = sub_grid_size
        super(GridImageSub, self).__init__(grid)

    @classmethod
    def from_mask(cls, mask, sub_grid_size):
        """ Given an image.Mask, compute the image analysis sub-grid using the center of every unmasked pixel.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the image sub-grid is computed for and the image's data grid.
        sub_grid_size : int
            The (sub_grid_size x sub_grid_size) of the sub-grid of each image pixel.
        """
        return GridImageSub(mask.compute_image_sub_grid(sub_grid_size), sub_grid_size)


class GridBlurring(Grid):

    def __init__(self, grid):
        """The blurring analysis grid, representing all pixels which are outside the image mask but will have a \
         fraction of their light blurred into the mask via PSF convolution.

        Parameters
        -----------
        grid : np.ndarray[blurring_pixels, 2]
            Array containing the blurring grid grids. The first elements map to an image pixel, and second to its \
            (x,y) arc second grids. E.g. the value grid[3,1] give the 4th blurring pixel's y coordinate.
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


class GridMapperSparse(object):
    """Analysis grid mappings between the sparse grid and image grid.
    """

    def __init__(self, sparse_to_image, image_to_sparse):

        self.sparse_to_image = sparse_to_image
        self.image_to_sparse = image_to_sparse

    @classmethod
    def from_mask(cls, mask, sparse_grid_size):
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

    def relocate_grid_coordinates_outside_border(self, grid):
        """For an input grid, return a grid where all coordinates outside the border are relocated to its edge."""

        self.polynomial_fit_to_border(grid)

        relocated_grid = np.zeros(grid.shape)

        for (i, coordinate) in enumerate(grid):
            relocated_grid[i] = self.relocated_coordinate(coordinate)

        return relocated_grid

    def relocate_sub_grid_coordinates_outside_border(self, grid, sub_grid):
        """For an input sub-grid, return a grid where all sub-coordinates outside the border are relocated to its edge.
        """

        # TODO : integrate these as functions into Grid and SubGrid, or pass in a Grid / SubGrid?

        self.polynomial_fit_to_border(grid)

        relocated_sub_grid = np.zeros(sub_grid.shape)

        for image_pixel in range(len(grid)):
            for (sub_pixel, sub_coordinate) in enumerate(sub_grid[image_pixel]):
                relocated_sub_grid[image_pixel, sub_pixel] = self.relocated_coordinate(sub_coordinate)

        return relocated_sub_grid

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