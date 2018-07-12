import numpy as np
from src.profiles import geometry_profiles


class CoordsCollection(object):

    def __init__(self, image, sub, blurring):
        """A collection of grids which contain the coordinates of an image. This includes the image's regular grid,
        sub-gri, blurring region, etc.

        Coordinate grids are passed through the ray-tracing module to set up the image, lens and source planes.
         
        Parameters
        -----------
        image : GridCoordsImage
            A grid of coordinates for the regular image grid.
        sub : GridCoordsImageSub
            A grid of coordinates for the sub-gridded image grid.
        blurring : GridCoordsBlurring
            A grid of coordinates for the blurring regions.
        """

        self.image = image
        self.sub = sub
        self.blurring = blurring

    def deflection_grids_for_galaxies(self, galaxies):
        """Compute the deflection angles of every grids (by integrating the mass profiles of the input galaxies)
        and set these up as a new collection of grids."""

        image = self.image.deflection_grid_for_galaxies(galaxies)
        sub = self.sub.deflection_grid_for_galaxies(galaxies)
        blurring = self.blurring.deflection_grid_for_galaxies(galaxies)

        return CoordsCollection(image, sub, blurring)

    def traced_grids_for_deflections(self, deflections):
        """Setup a new collection of grids by tracing their coordinates using a set of deflection angles."""
        image = self.image.ray_tracing_grid_for_deflections(deflections.image)
        sub = self.sub.ray_tracing_grid_for_deflections(deflections.sub)
        blurring = self.blurring.ray_tracing_grid_for_deflections(deflections.blurring)

        return CoordsCollection(image, sub, blurring)


class AbstractCoordinateGrid(np.ndarray):

    def __new__(cls, grid_coords):
        """Abstract base class for a set of grid coordinates, which store the arc-second coordinates of different \
        regions of an image. These are the coordinates used to perform ray-tracing and lensing analysis.

        Different grids are used to represent different regions of an image, thus controlling different aspects of the \
        analysis. For example, the image-sub coordinates are used to compute images on a uniform sub-grid, whereas \
        the blurring coordinates compute images in the areas which are outside the image-mask but close enough that \
        a fraction of their light is blurred into the masked region by the PSF.

        Each grid is stored as a structured array of coordinates, chosen for efficient ray-tracing \
        calculations. Coordinates are defined from the top-left corner, such that data_to_image in the top-left corner
        of an image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds. The image pixel indexes
        are also counted from the top-left.

        See *GridCoordsRegular* and *GridCoordsSub* for an illustration of each grid.
        """
        return np.array(grid_coords).view(cls)

    def deflection_grid_for_galaxies(self, galaxies):
        """ Setup a new image grid of coordinates, corresponding to the deflection angles computed from the mass \
        profile(s) of a set of galaxies at the image grid's coordinates.

        galaxies : [galaxy.Galaxy]
            The list of galaxies whose mass profiles are used to compute the deflection angles at the grid coordinates.
        """
        return self.new_from_array(self.deflections_on_grid(galaxies))

    def ray_tracing_grid_for_deflections(self, grid_deflections):
        """ Setup a new image grid of coordinates, by tracing the grid's coordinates using a set of \
        deflection angles which are also defined on the image-grid.

        Parameters
        -----------
        grid_deflections : GridCoordsImage
            The grid of deflection angles used to perform the ray-tracing.
        """
        return self.new_from_array(np.subtract(self, grid_deflections))

    def deflections_on_grid(self, galaxies):
        """Compute the intensity for each coordinate on the sub-grid, using the mass-profile(s) of a set of galaxies.

        Deflection angles are not averaged over a sub-pixel. Instead, the individual coordinates at each sub-pixel \
        are used to trace coordinates to the next plane.

        Parameters
        -----------
        galaxies : [galaxy.Galaxy]
            The list of galaxies whose mass profiles are used to compute the deflection angles at the grid coordinates.
        """
        return sum(map(lambda galaxy: galaxy.deflections_from_coordinate_grid(self), galaxies))

    def new_from_array(self, array):
        return self.__class__(array)


class CoordinateGrid(AbstractCoordinateGrid):
    """Abstract class for a regular grid of coordinates. On a regular grid, each pixel's arc-second coordinates \
    are represented by the value at the centre of the pixel.

    Coordinates are defined from the top-left corner, such that data_to_image in the top-left corner of an \
    image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds. The image pixel indexes are \
    also counted from the top-left.

    A regular *grid_coords* is a NumPy array of image_shape [image_pixels, 2]. Therefore, the first element maps \
    to the image pixel index, and second element to its (x,y) arc second coordinates. For example, the value \
    [3,1] gives the 4th image pixel's y coordinate.

    Below is a visual illustration of a regular grid, where a total of 10 data_to_image are unmasked and therefore \
    included in the grid.

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an example image.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from analysis)
    |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in analysis)
    |x|x|x|o|o|o|o|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    This image pixel index's will come out like this (and the direction of arc-second coordinates is highlighted
    around the image.

    pixel_scale = 1.0"

    <--- -ve  x  +ve -->

    |x|x|x|x|x|x|x|x|x|x|  ^   grid_coords[0] = [-0.5,  1.5]
    |x|x|x|x|x|x|x|x|x|x|  |   grid_coords[1] = [ 0.5,  1.5]
    |x|x|x|x|x|x|x|x|x|x|  |   grid_coords[2] = [-1.5,  0.5]
    |x|x|x|x|0|1|x|x|x|x| +ve  grid_coords[3] = [-0.5,  0.5]
    |x|x|x|2|3|4|5|x|x|x|  y   grid_coords[4] = [ 0.5,  0.5]
    |x|x|x|6|7|8|9|x|x|x| -ve  grid_coords[5] = [ 1.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x|  |   grid_coords[6] = [-1.5, -0.5]
    |x|x|x|x|x|x|x|x|x|x|  |   grid_coords[7] = [-0.5, -0.5]
    |x|x|x|x|x|x|x|x|x|x| \/   grid_coords[8] = [ 0.5, -0.5]
    |x|x|x|x|x|x|x|x|x|x|      grid_coords[9] = [ 1.5, -0.5]
    """

    def intensities_via_grid(self, galaxies):
        """Compute the intensity for each coordinate on the grid, using the light-profile(s) of a set of galaxies.

        Parameters
        -----------
        galaxies : [galaxy.Galaxy]
            The list of galaxies whose light profiles are used to compute the intensity at grid coordinate.
        """
        return sum(map(lambda galaxy: galaxy.intensity_from_grid(self), galaxies))


class SubCoordinateGrid(AbstractCoordinateGrid):
    """Abstract class for a sub of coordinates. On a sub-grid, each pixel is sub-gridded into a uniform grid of
     sub-coordinates, which are used to perform over-sampling in the lens analysis.

    Coordinates are defined from the top-left corner, such that data_to_image in the top-left corner of an
    image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds. The image pixel indexes are
    also counted from the top-left.

    A sub *grid_coords* is a NumPy array of image_shape [image_pixels, sub_grid_pixels, 2]. Therefore, the first
    element maps to the image pixel index, the second element to the sub-pixel index and third element to that
    sub pixel's (x,y) arc second coordinates. For example, the value [3, 6, 1] gives the 4th image pixel's
    7th sub-pixel's y coordinate.

    Below is a visual illustration of a sub grid. Like the regular grid, the indexing of each sub-pixel goes from
    the top-left corner. In contrast to the regular grid above, our illustration below restricts the mask to just
    2 data_to_image, to keep the illustration brief.

    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     This is an example image.Mask, where:
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from analysis)
    |x|x|x|x|o|o|x|x|x|x|     o = False (Pixel is not masked and included in analysis)
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|
    |x|x|x|x|x|x|x|x|x|x|

    Our regular-grid looks like it did before:

    pixel_scale = 1.0"

    <--- -ve  x  +ve -->

    |x|x|x|x|x|x|x|x|x|x|  ^
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x| +ve  grid_coords[0] = [-1.5,  0.5]
    |x|x|x|0|1|x|x|x|x|x|  y   grid_coords[1] = [-0.5,  0.5]
    |x|x|x|x|x|x|x|x|x|x| -ve
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x|  |
    |x|x|x|x|x|x|x|x|x|x| \/
    |x|x|x|x|x|x|x|x|x|x|

    However, we now go to each image-pixel and derive a sub-pixel grid for it. For example, for pixel 0,
    if *sub_grid_size=2*, we use a 2x2 sub-grid:

    Pixel 0 - (2x2):

           grid_coords[0,0] = [-1.66, 0.66]
    |0|1|  grid_coords[0,1] = [-1.33, 0.66]
    |2|3|  grid_coords[0,2] = [-1.66, 0.33]
           grid_coords[0,3] = [-1.33, 0.33]

    Now, we'd normally sub-grid all data_to_image using the same *sub_grid_size*, but for this illustration lets
    pretend we used a sub_grid_size of 3x3 for pixel 1:

             grid_coords[0,0] = [-0.75, 0.75]
             grid_coords[0,1] = [-0.5,  0.75]
             grid_coords[0,2] = [-0.25, 0.75]
    |0|1|2|  grid_coords[0,3] = [-0.75,  0.5]
    |3|4|5|  grid_coords[0,4] = [-0.5,   0.5]
    |6|7|8|  grid_coords[0,5] = [-0.25,  0.5]
             grid_coords[0,6] = [-0.75, 0.25]
             grid_coords[0,7] = [-0.5,  0.25]
             grid_coords[0,8] = [-0.25, 0.25]

    """

    def __new__(cls, grid_coords, sub_grid_size):
        """
        Parameters
        -----------
        grid_coords : np.ndarray
            The coordinates of the sub-grid.
        sub_grid_size : int
            The (sub_grid_size x sub_grid_size) sub_grid_size of each sub-grid for each pixel.
        """
        grid = super(SubCoordinateGrid, cls).__new__(cls, grid_coords)
        grid.sub_grid_size = sub_grid_size
        return grid

    def intensities_via_grid(self, galaxies, mapping):
        """Compute the intensity for each coordinate on the grid, using the light-profile(s) of a set of galaxies.

        Parameters
        -----------
        mapping
        galaxies : [galaxy.Galaxy]
            The list of galaxies whose light profiles are used to compute the intensity at the grid coordinates.
        """
        sub_intensities = sum(map(lambda galaxy: galaxy.intensity_from_grid(self), galaxies))
        return mapping.map_data_sub_to_image(sub_intensities)

    def new_from_array(self, array):
        return __class__(array, self.sub_grid_size)


class GridMapping(object):

    def __init__(self, image_shape, image_pixels, data_to_image, sub_grid_size, sub_to_image, cluster=None):
        self.image_shape = image_shape
        self.image_pixels = image_pixels
        self.data_to_image = data_to_image
        self.sub_pixels = sub_to_image.shape[0]
        self.sub_grid_size = sub_grid_size
        self.sub_grid_fraction = (1.0 / sub_grid_size ** 2.0)
        self.sub_to_image = sub_to_image
        self.cluster = cluster

    def map_data_sub_to_image(self, data):
        return np.multiply(self.sub_grid_fraction, data.reshape(-1, self.sub_grid_size ** 2).sum(axis=1))

    def map_to_2d(self, grid_data):
        """Use mapper to map an input data-set from a *GridData* to its original 2D image.

        Parameters
        -----------
        grid_data : ndarray
            The grid-data which is mapped to its 2D image.
        """
        data_2d = np.zeros(self.image_shape)

        for (i, pixel) in enumerate(self.data_to_image):
            data_2d[pixel[0], pixel[1]] = grid_data[i]

        return data_2d


class GridBorder(geometry_profiles.Profile):

    # TODO : Could speed this up by only looking and relocating image data_to_image within a certain radius of the image
    # TODO : centre. This would add a central_pixels lists to the input.

    # TODO : Why is this a profile?

    def __init__(self, border_pixels, polynomial_degree=3, centre=(0.0, 0.0)):
        """ The border of a set of grid coordinates, which relocates coordinates outside of the border to its edge.

        This is required to ensure highly demagnified data_to_image in the centre of an image do not bias a source
        pixelization.

        Parameters
        ----------
        border_pixels : np.ndarray
            The the border source data_to_image, specified by their 1D index in *image_grid*.
        polynomial_degree : int
            The degree of the polynomial used to fit the source-plane border edge.
        """

        super(GridBorder, self).__init__(centre)

        self.border_pixels = border_pixels
        self.polynomial_degree = polynomial_degree
        self.centre = centre

        self.thetas = None
        self.radii = None
        self.polynomial = None

    def relocate_coordinates_outside_border(self, coordinates):
        """For an input set of coordinates, return a new set of coordinates where every coordinate outside the border
        is relocated to its edge.

        Parameters
        ----------
        coordinates : ndarray
            The coordinates which are to have border relocations take place.
        """

        self.polynomial_fit_to_border(coordinates)

        relocated_coordinates = np.zeros(coordinates.shape)

        for (i, coordinate) in enumerate(coordinates):
            relocated_coordinates[i] = self.relocated_coordinate(coordinate)

        return relocated_coordinates

    def relocate_sub_coordinates_outside_border(self, coordinates, sub_coordinates):
        """For an input sub-coordinates, return a coordinates where all sub-coordinates outside the border are relocated
        to its edge.
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
        coordinates : Union((float, float), ndarray)
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
         A move-factor defines how far a coordinate outside the source-plane setup_border_pixels must be moved in order
         to lie on it. PlaneCoordinates already within the setup_border_pixels return a move-factor of 1.0, signifying
         they are already within the setup_border_pixels.

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
