import numpy as np

from auto_lens.imaging import imaging
from auto_lens.profiles import geometry_profiles


class GridCoordsCollection(object):

    def __init__(self, image, sub=None, blurring=None):
        """A collection of grids which contain the coordinates of an image. This includes the image's regular grid, \
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

    @classmethod
    def from_mask(cls, mask, grid_size_sub=None, blurring_size=None):
        """Setup the collection of coordinate grids using an image mask.

        Parameters
        -----------
        mask : imaging.Mask
            A mask describing which pixels the coordinates are computed for and used to setup the collection of grids.
        grid_size_sub : int
            The (grid_size_sub x grid_size_sub) size of each sub-grid for each pixel, used by *GridCoordsImageSub*.
        blurring_size : (int, int)
           The size of the psf which defines the blurring region, used by *GridCoordsBlurring*.
        """

        image = GridCoordsImage.from_mask(mask)

        if grid_size_sub is None:
            sub = None
        elif grid_size_sub is not None:
            sub = GridCoordsImageSub.from_mask(mask, grid_size_sub)

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


class GridCoords(np.ndarray):

    def __new__(cls, grid_coords):
        """Abstract base class for a set of grid coordinates, which store the arc-second coordinates of different \
        regions of an image. These are the coordinates used to perform ray-tracing and lensing analysis.

        Different grids are used to represent different regions of an image, thus controlling different aspects of the \
        analysis. For example, the image-sub coordinates are used to compute images on a uniform sub-grid, whereas \
        the blurring coordinates compute images in the areas which are outside the image-mask but close enough that \
        a fraction of their light is blurred into the masked region by the PSF.

        Each grid is stored as a structured array of coordinates, chosen for efficient ray-tracing \
        calculations. Coordinates are defined from the top-left corner, such that pixels in the top-left corner of an \
        image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds. The image pixel indexes are \
        also counted from the top-left.

        See *GridCoordsRegular* and *GridCoordsSub* for an illustration of each grid.
        """
        return np.array(grid_coords).view(cls)


class GridCoordsRegular(GridCoords):

    def __new__(cls, grid_coords):
        """Abstract class for a regular grid of coordinates. On a regular grid, each pixel's arc-second coordinates \
        are represented by the value at the centre of the pixel.

        Coordinates are defined from the top-left corner, such that pixels in the top-left corner of an \
        image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds. The image pixel indexes are \
        also counted from the top-left.

        A regular *grid_coords* is a NumPy array of dimensions [image_pixels, 2]. Therefore, the first element maps \
        to the image pixel index, and second element to its (x,y) arc second coordinates. For example, the value \
        [3,1] gives the 4th image pixel's y coordinate.

        Below is a visual illustration of a regular grid, where a total of 10 pixels are unmasked and therefore \
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

        This image pixel index's will come out like this (and the direction of arc-second coordinates is highlighted \
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

        Parameters
        -----------
        grid_coords : np.ndarray
            The coordinates of the regular grid.
        """
        return super(GridCoordsRegular, cls).__new__(cls, grid_coords)

    def intensities_via_grid(self, galaxies):
        """Compute the intensity for each coordinate on the grid, using the light-profile(s) of a set of galaxies.

        Parameters
        -----------
        galaxies : [galaxy.Galaxy]
            The list of galaxies whose light profiles are used to compute the intensity at grid coordinate.
        """
        return sum(map(lambda galaxy : self.evaluate_func_on_grid(func=galaxy.intensity_at_coordinates,
                                                                  output_shape=self.shape[0]), galaxies))

    def deflections_on_grid(self, galaxies):
        """Compute the deflection angle for each coordinate on the grid, using the mass-profile(s) of a set of \
         galaxies.

        Parameters
        -----------
        galaxies : [galaxy.Galaxy]
            The list of galaxies whose light profiles are used to compute the intensity at grid coordinate.
        """
        return sum(map(lambda galaxy : self.evaluate_func_on_grid(func=galaxy.deflections_at_coordinates,
                                                                  output_shape=self.shape), galaxies))

    def evaluate_func_on_grid(self, func, output_shape):
        """Compute a set of values (intensities, surface densities, potentials or deflections angles) for a light or \
        mass profile for each coordinate on a regular grid.

        NOTES
        ----------

        The output shape is included as an input because:

        - For deflection angles, the output array's shape is the same as the grid (e.g. the input grid is \
        [image_pixels, 2] and output grid is [image_pixels, 2]).

        - For intensities, surface-densities and potentials, the output array's shape loses the second dimension \
        (e.g. the input grid is [image_pixels, 2] and output grid is [image_pixels]).

        Parameters
        -----------
        func : func
            The *LightProfile* or *MassProfile* calculation function (e.g. intensity_at_coordinates).
        output_shape : (int, int)
            The output dimensions of the evaluated values.

        """
        grid_values = np.zeros(output_shape)

        for pixel_no, coordinate in enumerate(self):
            grid_values[pixel_no] = func(coordinates=coordinate)

        return grid_values


class GridCoordsSub(GridCoords):

    def __new__(cls, grid_coords, grid_size_sub):
        """Abstract class for a sub of coordinates. On a sub-grid, each pixel's is sub-gridded into a uniform grid of \
         sub-coordinates, which are used to perform over-sampling in the lens analysis.

        Coordinates are defined from the top-left corner, such that pixels in the top-left corner of an \
        image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds. The image pixel indexes are \
        also counted from the top-left.

        A sub *grid_coords* is a NumPy array of dimensions [image_pixels, sub_grid_pixels, 2]. Therefore, the first \
        element maps to the image pixel index, the second element to the sub-pixel index and third element to that \
        sub pixel's (x,y) arc second coordinates. For example, the value [3, 6, 1] gives the 4th image pixel's \
        7th sub-pixel's y coordinate.

        Below is a visual illustration of a sub grid. Like the regular grid, the indexing of each sub-pixel goes from \
        the top-left corner. In contrast to the regular grid above, our illustration below restricts the mask to just \
        2 pixels, to keep the illustration brief.

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
        if *grid_size_sub=2*, we use a 2x2 sub-grid:

        Pixel 0 - (2x2):

               grid_coords[0,0] = [-1.66, 0.66]
        |0|1|  grid_coords[0,1] = [-1.33, 0.66]
        |2|3|  grid_coords[0,2] = [-1.66, 0.33]
               grid_coords[0,3] = [-1.33, 0.33]

        Now, we'd normally sub-grid all pixels using the same *grid_size_sub*, but for this illustration lets pretend \
        we used a size of 3x3 for pixel 1:

                 grid_coords[0,0] = [-0.75, 0.75]
                 grid_coords[0,1] = [-0.5,  0.75]
                 grid_coords[0,2] = [-0.25, 0.75]
        |0|1|2|  grid_coords[0,3] = [-0.75,  0.5]
        |3|4|5|  grid_coords[0,4] = [-0.5,   0.5]
        |6|7|8|  grid_coords[0,5] = [-0.25,  0.5]
                 grid_coords[0,6] = [-0.75, 0.25]
                 grid_coords[0,7] = [-0.5,  0.25]
                 grid_coords[0,8] = [-0.25, 0.25]

        Parameters
        -----------
        grid_coords : np.ndarray
            The coordinates of the sub-grid.
        grid_size_sub : int
            The (grid_size_sub x grid_size_sub) size of each sub-grid for each pixel.
        """
        coords = super(GridCoordsSub, cls).__new__(cls, grid_coords)
        coords.grid_size_sub = grid_size_sub
        coords.grid_size_sub_squared = grid_size_sub ** 2.0
        return coords

    def intensities_via_grid(self, galaxies):
        """Compute the intensity for each coordinate on the sub-grid, using the light-profile(s) of a set of galaxies.

        For each sub-pixel, after computing the intensities at each sub coordinate, the mean is taken to compute \
        the overall intensity of that pixel.

        Parameters
        -----------
        galaxies : [galaxy.Galaxy]
            The list of galaxies whose light profiles are used to compute the intensity at the grid coordinates.
        """

        sub_intensities = sum(map(lambda galaxy : self.evaluate_func_on_grid(func=galaxy.intensity_at_coordinates,
                                                               output_shape=self.shape[0:2]), galaxies))

        intensities = np.zeros(self.shape[0])

        for pixel_no, intensities_sub_pixel in enumerate(sub_intensities):
            intensities[pixel_no] = np.sum(intensities_sub_pixel) / self.grid_size_sub_squared

        return intensities

    def deflections_on_grid(self, galaxies):
        """Compute the intensity for each coordinate on the sub-grid, using the mass-profile(s) of a set of galaxies.

        Deflection angles are not averaged over a sub-pixel. Instead, the individual coordinates at each sub-pixel \
        are used to trace coordinates to the next plane.

        Parameters
        -----------
        galaxies : [galaxy.Galaxy]
            The list of galaxies whose mass profiles are used to compute the deflection angles at the grid coordinates.
        """
        return sum(map(lambda galaxy : self.evaluate_func_on_grid(func=galaxy.deflections_at_coordinates,
                                                                  output_shape=self.shape), galaxies))

    def evaluate_func_on_grid(self, func, output_shape):
        """Compute a set of values (e.g. intensities or deflections angles) for a light or mass profile, at the set of \
        coordinates defined by a sub-grid_coords.
        """

        sub_grid_values = np.zeros(output_shape)

        for pixel_no, pixel_sub_grid in enumerate(self):
            for sub_pixel_no, sub_coordinate in enumerate(pixel_sub_grid):
                sub_grid_values[pixel_no, sub_pixel_no] = func(coordinates=sub_coordinate)

        return sub_grid_values


class GridCoordsImage(GridCoordsRegular):

    def __new__(cls, grid_coords):
        """The coordinates of each pixel in an image, stored using a regular grid.

        See *GridCoordsRegular* for more details.

        Parameters
        -----------
        grid_coords : np.ndarray
            The coordinates of the image, on a regular grid.
        """
        return super(GridCoordsImage, cls).__new__(cls, grid_coords)

    @classmethod
    def from_mask(cls, mask):
        """ Given an image.Mask, setup the grid of image coordinates using the center of every unmasked pixel.

        Parameters
        ----------
        mask : imaging.Mask
            A mask describing which pixels the coordinates are computed for to setup the image grid.
        """
        return GridCoordsImage(mask.compute_grid_coords_image())

    def setup_deflections_grid(self, galaxies):
        """ Setup a new image grid of coordinates, corresponding to the deflection angles computed from the mass \
        profile(s) of a set of galaxies at the image grid's coordinates.

        galaxies : [galaxy.Galaxy]
            The list of galaxies whose mass profiles are used to compute the deflection angles at the grid coordinates.
        """
        return GridCoordsImage(self.deflections_on_grid(galaxies))

    def setup_traced_grid(self, grid_deflections):
        """ Setup a new image grid of coordinates, by tracing the grid's coordinates using a set of \
        deflection angles which are also defined on the image-grid.

        Parameters
        -----------
        grid_deflections : GridCoordsImage
            The grid of deflection angles used to perform the ray-tracing.
        """
        return GridCoordsImage(np.subtract(self, grid_deflections))


class GridCoordsImageSub(GridCoordsSub):

    def __new__(cls, grid_coords, grid_size_sub):
        """The sub-coordinates of each pixel in an image, stored using a sub-grid.

        Parameters
        -----------
        grid_coords : np.ndarray
            The coordinates of the image sub-grid, on a sub-grid.
        grid_size_sub : int
            The (grid_size_sub x grid_size_sub) of the sub-grid_coords of each image pixel.
        """
        return super(GridCoordsImageSub, cls).__new__(cls, grid_coords, grid_size_sub)

    @classmethod
    def from_mask(cls, mask, grid_size_sub):
        """ Given an image.Mask, compute the image sub-grid_coordinates by sub-gridding every unmasked pixel around \
        its center.

        Parameters
        mask : imaging.Mask
            A mask describing which pixels the sub-coordinates are computed for to setup the image sub-grid.
        grid_size_sub : int
            The (grid_size_sub x grid_size_sub) of the sub-grid_coords of each image pixel.
        """
        return GridCoordsImageSub(mask.compute_grid_coords_image_sub(grid_size_sub), grid_size_sub)

    def setup_deflections_grid(self, galaxies):
        """ Setup a new image sub-grid of coordinates, corresponding to the deflection angles computed from the mass \
        profile(s) of a set of galaxies at the image sub-grid's coordinates.

        galaxies : [galaxy.Galaxy]
            The list of galaxies whose mass profiles are used to compute the deflection angles at the sub-grid \
            coordinates.
        """
        return GridCoordsImageSub(self.deflections_on_grid(galaxies), self.grid_size_sub)

    def setup_traced_grid(self, grid_deflections):
        """ Setup a new image sub-grid of coordinates, by tracing the sub-grid's coordinates using a set of \
        deflection angles which are also defined on the image sub-grid.

        Parameters
        -----------
        grid_deflections : GridCoordsImage
            The grid of deflection angles used to perform the ray-tracing.
        """
        return GridCoordsImageSub(np.subtract(self, grid_deflections), self.grid_size_sub)


class GridCoordsBlurring(GridCoordsRegular):

    def __new__(cls, grid_coords):
        """ The coordinates of each blurring pixel in an image, stored using a regular-grid. The blurring grid \
        contains all pixels which are outside the mask have a fraction of their light blurred into the mask via \
        PSF convolution.

        Parameters
        -----------
        grid_coords : np.ndarray
            The coordinates of the blurring regions, on a regular grid.
        """
        return super(GridCoordsBlurring, cls).__new__(cls, grid_coords)

    @classmethod
    def from_mask(cls, mask, psf_size):
        """ Given an image.Mask, compute the blurring coordinates grid_coords by locating all pixels which are \
        within the psf size of the mask.

        Parameters
        ----------
        mask : imaging.Mask
            A mask describing which pixels the image coordinates are computed for, and therefore from which the \
            blurring regions can be computed.
        psf_size : (int, int)
           The size of the psf which defines the blurring region (e.g. the pixel_dimensions of the PSF)
        """
        if psf_size[0] % 2 == 0 or psf_size[1] % 2 == 0:
            raise imaging.MaskException("psf_size of exterior region must be odd")

        return GridCoordsBlurring(mask.compute_grid_coords_blurring(psf_size))

    def setup_deflections_grid(self, galaxies):
        """ Setup a new blurring grid of coordinates, corresponding to the deflection angles computed from the mass \
        profile(s) of a set of galaxies at the blurring grid's coordinates.

        galaxies : [galaxy.Galaxy]
            The list of galaxies whose mass profiles are used to compute the deflection angles at the grid coordinates.
        """
        return GridCoordsBlurring(self.deflections_on_grid(galaxies))

    def setup_traced_grid(self, grid_deflections):
        """ Setup a new blurring grid of coordinates, by tracing the grid's coordinates using a set of \
        deflection angles which are also defined on the blurring-grid.

        Parameters
        -----------
        grid_deflections : GridCoordsImage
            The grid of deflection angles used to perform the ray-tracing.
        """
        return GridCoordsBlurring(np.subtract(self, grid_deflections))


# TODO : We'll probably end up splitting 'GridData' into different data-types .e.g 'GridImage', 'GridNoise',
# TODO : 'GridImageLensSubtracted', etc.


class GridDataCollection(object):

    def __init__(self, image, noise, exposure_time, psf):
        """A collection of grids which contain the data (image, noise, exposure times, psf).

        Parameters
        -----------
        image : GridData
            A data-grid of the observed image fluxes (electrons per second)
        noise : GridData
            A data-grid of the observed image noise estimates (standard deviations, electrons per second)
        exposure_time : GridData
            A data-grid of the exposure time in each pixel (seconds)
        psf : imaging.PSF
            The 2D Point Spread Function (PSF).
        """
        self.image = image
        self.noise = noise
        self.exposure_time = exposure_time
        self.psf = psf


class GridData(np.ndarray):

    def __new__(cls, grid_data):
        """The grid storing the value in each unmasked pixel of a data-set (e.g. an image, noise, exposure times, etc.).

        Data values are defined from the top-left corner, such that pixels in the top-left corner of an \
        image (e.g. [0,0]) have the lowest index value. Therefore, the *grid_data* is a NumPy array of dimensions \
        [image_pixels], where each element maps to its corresponding image pixel index. For example, the value \
        [3] gives the 4th pixel's data value.

        Below is a visual illustration of a data-grid, where a total of 10 pixels are unmasked and therefore \
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

        Now lets pretend these are the data values of this grid:

        |1|6|8|3|4|5|7|4|3|2|
        |7|3|6|4|8|1|2|2|4|3|       
        |6|0|7|4|1|0|6|6|3|0|
        |5|7|6|0|2|8|4|4|2|0|
        |3|3|3|9|3|4|6|3|1|0|
        |4|2|4|6|7|1|3|2|2|2|
        |5|3|5|9|7|2|2|2|2|3|
        |6|4|5|9|5|3|1|4|3|6|
        |6|5|6|9|3|4|2|0|7|4|
        |3|6|7|6|2|5|4|0|8|2|
        
        Lets extract specifically the data which is unmasked and look at our grid_data:
        
        |x|x|x|x|x|x|x|x|x|x|   grid_data[0] = 2
        |x|x|x|x|x|x|x|x|x|x|   grid_data[1] = 8
        |x|x|x|x|x|x|x|x|x|x|   grid_data[2] = 9
        |x|x|x|x|2|8|x|x|x|x|   grid_data[3] = 3
        |x|x|x|9|3|4|6|x|x|x|   grid_data[4] = 4
        |x|x|x|6|7|1|3|x|x|x|   grid_data[5] = 6
        |x|x|x|x|x|x|x|x|x|x|   grid_data[6] = 6
        |x|x|x|x|x|x|x|x|x|x|   grid_data[7] = 7
        |x|x|x|x|x|x|x|x|x|x|   grid_data[8] = 1
        |x|x|x|x|x|x|x|x|x|x|   grid_data[9] = 3

        Parameters
        -----------
        grid_data : np.ndarray
            The data-values in the unmasked pixels of a data-set (e.g. an image, noise, exposure times).

        Notes
        ----------

        The *GridData* and *GridCoords* used in an analysis must correspond to the same masked region of an image. \
        The easiest way to ensure this is to generate them all from the same mask.

        """
        return np.array(grid_data).view(cls)

    @classmethod
    def from_mask(cls, data, mask):
        """ Given an image.Mask, setup the data-grid using the every unmasked pixel.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the data-grid is computed for.
        """
        return GridData(mask.compute_grid_data(data))


class GridMapperCollection(object):

    def __init__(self, data_to_2d, clustering=None):
        """A collection of mappers, which map between data on different grids.

        Parameters
        -----------
        data_to_2d : GridMapperDataTo2D
            Mapper between 1D *GridData* and its 2D image coordinates.
        clustering : GridMapperClustering
            Mapper between image pixels and the clustering grid pixels.
        """

        self.data_to_2d = data_to_2d
        self.clustering = clustering


class GridMapperDataTo2D(np.ndarray):

    def __new__(cls, dimensions, data_to_2d):
        """A grid which maps every value of the *GridData* to its 2D pixel, used to rebuild 1D data-grids to 2D \
        for visualization.

        This also stores the data's original 2D dimensions, so that the rebuilt image is at the original size. \

        The mapper is a NumPy array of dimensions [image_pixels, 2]. Therefore, the first element maps to the \
        image pixel index, and second element to its (x,y) pixel coordinates. For example, the value [3,1] gives \
        the 4th image pixel's y pixel.

        Below is a visual illustration, where a total of 10 pixels are unmasked and therefore \
        included in the mapper.

             0 1 2 3 4 5 6 7 8 9

        0   |x|x|x|x|x|x|x|x|x|x|
        1   |x|x|x|x|x|x|x|x|x|x|     This is an example image.Mask, where:
        2   |x|x|x|x|x|x|x|x|x|x|
        3   |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from analysis)
        4   |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in analysis)
        5   |x|x|x|o|o|o|o|x|x|x|
        6   |x|x|x|x|x|x|x|x|x|x|
        7   |x|x|x|x|x|x|x|x|x|x|
        8   |x|x|x|x|x|x|x|x|x|x|
        9   |x|x|x|x|x|x|x|x|x|x|

        Remembering that we count pixels rightwards from the top left corner (see *GridRegular), the data_to_2d \
        vector will read:

        data_to_2d[0] = [3,4]
        data_to_2d[1] = [3,5]
        data_to_2d[2] = [4,3]
        data_to_2d[3] = [4,4]
        data_to_2d[4] = [4,5]
        data_to_2d[5] = [4,6]
        data_to_2d[6] = [5,3]
        data_to_2d[7] = [5,4]
        data_to_2d[8] = [5,5]
        data_to_2d[9] = [5,6]

        Parameters
        -----------
        dimensions : (int, int)
            The 2D dimensions of the data's original image.
        data_to_2d : ndarray
            Numpy array containing the pixel coordinates of each data point.
        """
        mapper =  np.array(data_to_2d).view(cls)
        mapper.dimensions = dimensions
        return mapper

    @classmethod
    def from_mask(cls, mask):
        """Using an image.Mask, setup data to 2d mapper.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the unmasked pixels of the data grid.
        """
        return GridMapperDataTo2D(dimensions=mask.pixel_dimensions, data_to_2d=mask.compute_grid_mapper_data_to_2d())

    def map_to_2d(self, grid_data):
        """Use mapper to map an input data-set from a *GridData* to its original 2D image.

        Parameters
        -----------

        grid_data : GridData
            The grid-data which is mapped to its 2D image.
        """
        data_2d = np.zeros(self.dimensions)

        for (i, pixel) in enumerate(self):
            data_2d[pixel[0], pixel[1]] = grid_data[i]

        return data_2d


class GridMapperClustering(object):

    def __init__(self, clustering_to_image, image_to_clustering):
        """ The KMeans clustering used to derive an amorphous pixeliation uses a set of image-grid coordinates. For \
        high resolution imaging, the large number of coordinates makes KMeans clustering (unfeasibly) slow.

        Therefore, for efficiency, we define a 'clustering-grid', which is a sparsely sampled set of image-grid \
        coordinates used by the KMeans algorithm instead. However, we don't need the actual coordinates of this \
        clustering grid (as they are already calculated for the image-grid). Instead, we just need a mapper between \
        clustering-pixels and image-pixels.

        Thus, the *clustering_to_image* attribute maps every pixel on the clustering grid to its closest image pixel \
        (via the image pixel's 1D index). This is used before the KMeans clustering algorithm, to extract the sub-set \
        of coordinates that the algorithm uses.

        By giving the KMeans algorithm only clustering-grid coordinates, it will only tell us the mappings between \
        source-pixels and clustering-pixels. However, to perform the source reconstruction, we need to know all of the \
        mappings between source pixels and image pixels / sub-image pixels.This would require a (computationally \
        expensive) nearest-neighbor search (over all clustering pixels and image / sub pixels) to calculate. The \
        calculation can be sped-up by using the attribute *image_to_clustering*, which maps every image-pixel to its \
        closest pixel on the clustering grid (see *pixelization.sub_coordinates_to_source_pixels_via_sparse_pairs*).
        """

        self.clustering_to_image = clustering_to_image
        self.image_to_clustering = image_to_clustering

    @classmethod
    def from_mask(cls, mask, cluster_grid_size):
        """ Given an image.Mask, compute the clustering mapper of the image by inputting its size and finding \
        all image pixels which are on its sparsely defined mask.

        Parameters
        ----------
        mask : imaging.Mask
            The image mask containing the pixels the blurring grid_coords is computed for and the image's data grid_coords.
        """
        sparse_to_image, image_to_sparse = mask.compute_grid_mapper_sparse(cluster_grid_size)
        return GridMapperClustering(sparse_to_image, image_to_sparse)


class GridBorder(geometry_profiles.Profile):

    # TODO : Could speed this up by only looking and relocating image pixels within a certain radius of the image
    # TODO : centre. This would add a central_pixels lists to the input.

    def __init__(self, border_pixels, polynomial_degree=3, centre=(0.0, 0.0)):
        """ The border of a set of grid coordinates, which relocates coordinates outside of the border to its edge.

        This is required to ensure highly demagnified pixels in the centre of an image do not bias a source \
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
        """For an input set of coordinates, return a new set of coordinates where every coordinate outside the border \
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