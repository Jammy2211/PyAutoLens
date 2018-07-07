import numpy as np
from src import exc
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

    @classmethod
    def from_mask(cls, mask, grid_size_sub, blurring_shape):
        """Setup the collection of coordinate grids using an image mask.

        Parameters
        -----------
        mask : mask.Mask
            A mask describing which data_to_pixel the coordinates are computed for and used to setup the collection of
            grids.
        grid_size_sub : int
            The (grid_size_sub x grid_size_sub) size of each sub-grid for each pixel, used by *GridCoordsImageSub*.
        blurring_shape : (int, int)
           The size of the psf which defines the blurring region, used by *GridCoordsBlurring*.
        """

        image = GridCoordsImage.from_mask(mask)
        sub = GridCoordsImageSub.from_mask(mask, grid_size_sub)
        blurring = GridCoordsBlurring.from_mask(mask, blurring_shape)

        return CoordsCollection(image, sub, blurring)

    def deflection_grids_for_galaxies(self, galaxies):
        """Compute the deflection angles of every grids (by integrating the mass profiles of the input galaxies)
        and set these up as a new collection of grids."""

        image = self.image.setup_deflections_grid(galaxies)
        sub = self.sub.setup_deflections_grid(galaxies)
        blurring = self.blurring.setup_deflections_grid(galaxies)

        return CoordsCollection(image, sub, blurring)

    def traced_grids_for_deflections(self, deflections):
        """Setup a new collection of grids by tracing their coordinates using a set of deflection angles."""
        image = self.image.setup_traced_grid(deflections.image)
        sub = self.sub.setup_traced_grid(deflections.sub)
        blurring = self.blurring.setup_traced_grid(deflections.blurring)

        return CoordsCollection(image, sub, blurring)


class GridCoords(np.ndarray):

    def __new__(cls, coords):
        """Abstract base class for a set of grid coordinates, which store the arc-second coordinates of different \
        regions of an image. These are the coordinates used to perform ray-tracing and lensing analysis.

        Different grids are used to represent different regions of an image, thus controlling different aspects of the \
        analysis. For example, the image-sub coordinates are used to compute images on a uniform sub-grid, whereas \
        the blurring coordinates compute images in the areas which are outside the image-mask but close enough that \
        a fraction of their light is blurred into the masked region by the PSF.

        Each grid is stored as a structured array of coordinates, chosen for efficient ray-tracing \
        calculations. Coordinates are defined from the top-left corner, such that data_to_pixel in the top-left corner
        of an image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds. The image pixel indexes
        are also counted from the top-left.

        See *GridCoordsRegular* and *GridCoordsSub* for an illustration of each grid.
        """
        return np.array(coords).view(cls)

    def deflections_on_grid(self, galaxies):
        """Compute the deflection angle for each coordinate on the grid, using the mass-profile(s) of a set of \
         galaxies.

        Parameters
        -----------
        galaxies : [galaxy.Galaxy]
            The list of galaxies whose light profiles are used to compute the intensity at grid coordinate.
        """
        return sum(map(lambda galaxy: self.evaluate_func_on_grid(func=galaxy.deflections_at_coordinates,
                                                                 output_shape=self.shape), galaxies))

    # TODO : I'll let you make this efficienctly call the profiles

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
            The output dimensions_2d of the evaluated values.

        """
        grid_values = np.zeros(output_shape)

        for pixel_no, coordinate in enumerate(self):
            grid_values[pixel_no] = func(coordinates=coordinate)

        return grid_values


class GridCoordsRegular(GridCoords):
    """Abstract class for a regular grid of coordinates. On a regular grid, each pixel's arc-second coordinates \
    are represented by the value at the centre of the pixel.

    Coordinates are defined from the top-left corner, such that data_to_pixel in the top-left corner of an \
    image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds. The image pixel indexes are \
    also counted from the top-left.

    A regular *grid_coords* is a NumPy array of dimensions_2d [image_pixels, 2]. Therefore, the first element maps \
    to the image pixel index, and second element to its (x,y) arc second coordinates. For example, the value \
    [3,1] gives the 4th image pixel's y coordinate.

    Below is a visual illustration of a regular grid, where a total of 10 data_to_pixel are unmasked and therefore \
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
        return sum(map(lambda galaxy: self.evaluate_func_on_grid(func=galaxy.intensity_at_coordinates,
                                                                 output_shape=self.shape[0]), galaxies))


class GridCoordsSub(GridCoords):

    def __new__(cls, coords, grid_size_sub, sub_to_image, image_pixels):
        """Abstract class for a sub of coordinates. On a sub-grid, each pixel is sub-gridded into a uniform grid of
         sub-coordinates, which are used to perform over-sampling in the lens analysis.

        Coordinates are defined from the top-left corner, such that data_to_pixel in the top-left corner of an
        image (e.g. [0,0]) have a negative x-value and positive y-value in arc seconds. The image pixel indexes are
        also counted from the top-left.

        A sub *grid_coords* is a NumPy array of dimensions_2d [image_pixels, sub_grid_pixels, 2]. Therefore, the first
        element maps to the image pixel index, the second element to the sub-pixel index and third element to that
        sub pixel's (x,y) arc second coordinates. For example, the value [3, 6, 1] gives the 4th image pixel's
        7th sub-pixel's y coordinate.

        Below is a visual illustration of a sub grid. Like the regular grid, the indexing of each sub-pixel goes from
        the top-left corner. In contrast to the regular grid above, our illustration below restricts the mask to just
        2 data_to_pixel, to keep the illustration brief.

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

        Now, we'd normally sub-grid all data_to_pixel using the same *grid_size_sub*, but for this illustration lets
        pretend we used a size of 3x3 for pixel 1:

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
        coords : np.ndarray
            The coordinates of the sub-grid.
        grid_size_sub : int
            The (grid_size_sub x grid_size_sub) size of each sub-grid for each pixel.
        """
        coords = super(GridCoordsSub, cls).__new__(cls, coords)
        coords.image_pixels = image_pixels
        coords.sub_pixels = coords.shape[0]
        coords.grid_size_sub = grid_size_sub
        coords.grid_size_sub_squared = np.square(grid_size_sub)
        coords.sub_to_image = sub_to_image
        return coords

    def map_data_to_image_grid(self, data_sub):

        data_image = np.zeros((self.image_pixels))

        for sub_pixel in range(self.sub_pixels):
            data_image[self.sub_to_image[sub_pixel]] += data_sub[sub_pixel]

        return data_image / self.grid_size_sub_squared

    def intensities_via_grid(self, galaxies):
        """Compute the intensity for each coordinate on the grid, using the light-profile(s) of a set of galaxies.

        Parameters
        -----------
        galaxies : [galaxy.Galaxy]
            The list of galaxies whose light profiles are used to compute the intensity at grid coordinate.
        """
        sub_intensities = sum(map(lambda galaxy: self.evaluate_func_on_grid(func=galaxy.intensity_at_coordinates,
                                                                 output_shape=self.shape[0]), galaxies))


class GridCoordsImage(GridCoordsRegular):
    """The coordinates of each pixel in an image, stored using a regular grid.

    See *GridCoordsRegular* for more details.
    """

    @classmethod
    def from_mask(cls, mask):
        """ Given an image.Mask, setup the grid of image coordinates using the center of every unmasked pixel.

        Parameters
        ----------
        mask : mask.Mask
            A mask describing which data_to_pixel the coordinates are computed for to setup the image grid.
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
    """The sub-coordinates of each pixel in an image, stored using a sub-grid.
    """

    @classmethod
    def from_mask(cls, mask, grid_size_sub):
        """ Given an image.Mask, compute the image sub-grid_coordinates by sub-gridding every unmasked pixel around \
        its center.

        Parameters
        -----------
        mask : mask.Mask
            A mask describing which data_to_pixel the sub-coordinates are computed for to setup the image sub-grid.
        grid_size_sub : int
            The (grid_size_sub x grid_size_sub) of the sub-grid_coords of each image pixel.
        """
        return GridCoordsImageSub(mask.compute_grid_coords_image_sub(grid_size_sub), grid_size_sub,
                                  mask.compute_grid_sub_to_image(grid_size_sub), mask.pixels_in_mask)

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

    def __new__(cls, coords):
        """ The coordinates of each blurring pixel in an image, stored using a regular-grid. The blurring grid \
        contains all data_to_pixel which are outside the mask have a fraction of their light blurred into the mask via \
        PSF convolution.

        Parameters
        -----------
        coords : np.ndarray
            The coordinates of the blurring regions, on a regular grid.
        """
        return super(GridCoordsBlurring, cls).__new__(cls, coords)

    @classmethod
    def from_mask(cls, mask, psf_size):
        """ Given an image.Mask, compute the blurring coordinates grid_coords by locating all data_to_pixel which are \
        within the psf size of the mask.

        Parameters
        ----------
        mask : mask.Mask
            A mask describing which data_to_pixel the image coordinates are computed for, and therefore from which the \
            blurring regions can be computed.
        psf_size : (int, int)
           The size of the psf which defines the blurring region (e.g. the shape of the PSF)
        """
        if psf_size[0] % 2 == 0 or psf_size[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

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


class DataCollection(object):

    def __init__(self, image, noise, exposure_time):
        """A collection of grids which contain the data (image, noise, exposure times, psf).

        Parameters
        -----------
        image : GridData
            A data-grid of the observed image fluxes (electrons per second)
        noise : GridData
            A data-grid of the observed image noise estimates (standard deviations, electrons per second)
        exposure_time : GridData
            A data-grid of the exposure time in each pixel (seconds)
        """
        self.image = image
        self.noise = noise
        self.exposure_time = exposure_time

    @classmethod
    def from_mask(cls, mask, image, noise, exposure_time):
        """Setup the collection of data grids using a mask.

        Parameters
        -----------
        mask : mask.Mask
            A mask describing which data_to_pixel the coordinates are computed for and used to setup the collection of
            grids.
        image : imaging.Image
            A data-grid of the observed image fluxes (electrons per second)
        noise : imaging.Noise
            A data-grid of the observed image noise estimates (standard deviations, electrons per second)
        exposure_time : imaging.ExposureTime
            A data-grid of the exposure time in each pixel (seconds)
        """
        image = GridData.from_mask(image, mask)
        noise = GridData.from_mask(noise, mask)
        exposure_time = GridData.from_mask(exposure_time, mask)
        return DataCollection(image, noise, exposure_time)


class GridData(np.ndarray):

    def __new__(cls, data):
        """The grid storing the value in each unmasked pixel of a data-set (e.g. an image, noise, exposure times, etc.).

        Data values are defined from the top-left corner, such that data_to_pixel in the top-left corner of an \
        image (e.g. [0,0]) have the lowest index value. Therefore, the *grid_data* is a NumPy array of dimensions_2d \
        [image_pixels], where each element maps to its corresponding image pixel index. For example, the value \
        [3] gives the 4th pixel's data value.

        Below is a visual illustration of a data-grid, where a total of 10 data_to_pixel are unmasked and therefore \
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
        data : np.ndarray
            The data-values in the unmasked data_to_pixel of a data-set (e.g. an image, noise, exposure times).

        Notes
        ----------

        The *GridData* and *GridCoords* used in an analysis must correspond to the same masked region of an image.
        The easiest way to ensure this is to generate them all from the same mask.

        """
        return np.array(data).view(cls)

    @classmethod
    def from_mask(cls, data, mask):
        """ Given an image.Mask, setup the data-grid using the every unmasked pixel.

        Parameters
        ----------
        data
        mask : mask.Mask
            The image mask containing the data_to_pixel the data-grid is computed for.
        """
        return GridData(mask.compute_grid_data(data))


class MapperCollection(object):

    def __init__(self, data_to_pixel, clustering=None):
        """A collection of mappers, which map between data on different grids.

        Parameters
        -----------
        data_to_pixel : GridMapperDataToPixel
            Mapper between 1D image *GridData* and its 2D image coordinates.
        clustering : MapperCluster
            Mapper between image data_to_pixel and the clustering grid data_to_pixel.
        """

        self.data_to_pixel = data_to_pixel
        self.clustering = clustering

    @classmethod
    def from_mask(cls, mask, cluster_grid_size=None):
        """Setup the collection of grid mappers using an image mask.

        Parameters
        -----------
        cluster_grid_size
        mask : mask.Mask
            A mask describing which data_to_pixel the coordinates are computed for and used to setup the collection of
            grids.

        """

        image_to_pixel = GridMapperDataToPixel.from_mask(mask)
        clustering = MapperCluster.from_mask(mask, cluster_grid_size) if cluster_grid_size is not None else None

        return MapperCollection(image_to_pixel, clustering)


class GridMapperDataToPixel(np.ndarray):

    def __new__(cls, dimensions_2d, data_to_pixel):
        """A grid which maps every value of the *GridData* to its 2D pixel, used to rebuild 1D data-grids to 2D \
        for visualization.

        This also stores the data's original 2D dimensions_2d, so that the rebuilt image is at the original size. \

        The mapper is a NumPy array of dimensions_2d [image_pixels, 2]. Therefore, the first element maps to the \
        image pixel index, and second element to its (x,y) pixel coordinates. For example, the value [3,1] gives \
        the 4th image pixel's y pixel.

        Below is a visual illustration, where a total of 10 data_to_pixel are unmasked and therefore \
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

        Remembering that we count data_to_pixel rightwards from the top left corner (see *GridRegular),
        the data_to_pixel vector will read:

        data_to_pixel[0] = [3,4]
        data_to_pixel[1] = [3,5]
        data_to_pixel[2] = [4,3]
        data_to_pixel[3] = [4,4]
        data_to_pixel[4] = [4,5]
        data_to_pixel[5] = [4,6]
        data_to_pixel[6] = [5,3]
        data_to_pixel[7] = [5,4]
        data_to_pixel[8] = [5,5]
        data_to_pixel[9] = [5,6]

        Parameters
        -----------
        dimensions_2d : (int, int)
            The 2D dimensions_2d of the data's original image.
        data_to_pixel : ndarray
            Numpy array containing the pixel coordinates of each data point.
        """
        mapper = np.array(data_to_pixel).view(cls)
        mapper.dimensions_2d = dimensions_2d
        mapper.dimensions_1d = data_to_pixel.shape[0]
        return mapper

    @classmethod
    def from_mask(cls, mask):
        """Using an image.Mask, setup a data to 2d mapper.

        Parameters
        ----------
        mask : mask.Mask
            The image mask containing the unmasked data_to_pixel of the data grid.
        """
        return GridMapperDataToPixel(dimensions_2d=mask.shape,
                                     data_to_pixel=mask.compute_grid_mapper_data_to_pixel())

    def map_to_2d(self, grid_data):
        """Use mapper to map an input data-set from a *GridData* to its original 2D image.

        Parameters
        -----------
        grid_data : ndarray
            The grid-data which is mapped to its 2D image.
        """
        data_2d = np.zeros(self.dimensions_2d)

        for (i, pixel) in enumerate(self):
            data_2d[pixel[0], pixel[1]] = grid_data[i]

        return data_2d

    def map_to_1d(self, data_2d):
        """Use mapper to map a 2D image back to its *GridData* structure.

        Parameters
        -----------
        data_2d : ndarray
            The image which is to be mapped back to its *GridData* structure.
        """
        data_1d = np.zeros(self.dimensions_1d)

        for (i, pixel) in enumerate(self):
            data_1d[i] = data_2d[pixel[0], pixel[1]]

        return data_1d


class MapperCluster(object):

    def __init__(self, cluster_to_image, image_to_cluster):
        """ The KMeans clustering used to derive an amorphous pixeliation uses a set of image-grid coordinates. For \
        high resolution imaging, the large number of coordinates makes KMeans clustering (unfeasibly) slow.

        Therefore, for efficiency, we define a 'clustering-grid', which is a sparsely sampled set of image-grid \
        coordinates used by the KMeans algorithm instead. However, we don't need the actual coordinates of this \
        clustering grid (as they are already calculated for the image-grid). Instead, we just need a mapper between \
        clustering-data_to_pixel and image-data_to_pixel.

        Thus, the *cluster_to_image* attribute maps every pixel on the clustering grid to its closest image pixel \
        (via the image pixel's 1D index). This is used before the KMeans clustering algorithm, to extract the sub-set \
        of coordinates that the algorithm uses.

        By giving the KMeans algorithm only clustering-grid coordinates, it will only tell us the mappings between \
        source-data_to_pixel and clustering-data_to_pixel. However, to perform the source reconstruction, we need to
        know all of the mappings between source data_to_pixel and image data_to_pixel / sub-image data_to_pixel. This
        would require a (computationally expensive) nearest-neighbor search (over all clustering data_to_pixel and
        image / sub data_to_pixel) to calculate. The calculation can be sped-up by using the attribute
        *image_to_cluster*, which maps every image-pixel to its closest pixel on the clustering grid (see
        *pixelization.sub_coordinates_to_source_pixels_via_sparse_pairs*).
        """

        self.cluster_to_image = cluster_to_image
        self.image_to_cluster = image_to_cluster

    @classmethod
    def from_mask(cls, mask, cluster_grid_size):
        """ Given an image.Mask, compute the clustering mapper of the image by inputting its size and finding \
        all image data_to_pixel which are on its sparsely defined mask.

        Parameters
        ----------
        cluster_grid_size
        mask : mask.Mask
            The image mask containing the data_to_pixel the blurring grid_coords is computed for and the image's data
            grid_coords.
        """
        cluster_to_image, image_to_cluster = mask.compute_grid_mapper_sparse(cluster_grid_size)
        return MapperCluster(cluster_to_image, image_to_cluster)


class GridBorder(geometry_profiles.Profile):

    # TODO : Could speed this up by only looking and relocating image data_to_pixel within a certain radius of the image
    # TODO : centre. This would add a central_pixels lists to the input.

    def __init__(self, border_pixels, polynomial_degree=3, centre=(0.0, 0.0)):
        """ The border of a set of grid coordinates, which relocates coordinates outside of the border to its edge.

        This is required to ensure highly demagnified data_to_pixel in the centre of an image do not bias a source
        pixelization.

        Parameters
        ----------
        border_pixels : np.ndarray
            The the border source data_to_pixel, specified by their 1D index in *image_grid*.
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
