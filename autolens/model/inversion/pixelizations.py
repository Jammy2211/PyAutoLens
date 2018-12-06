import numpy as np
import scipy.spatial
import sklearn.cluster

from autolens import exc
from autolens.data.array import grids, scaled_array
from autolens.model.inversion import mappers
from autolens.model.inversion.util import pixelization_util


def setup_image_plane_pixelization_grid_from_galaxies_and_grids(galaxies, data_grids):
    """An image-plane pixelization is one where its pixel centres are computed by tracing a sparse grid of pixels from \
    the data's regular grid to other planes (e.g. the source-plane).

    Provided a galaxy has an image-plane pixelization, this function returns a new *DataGrids* instance where the \
    image-plane pixelization's sparse grid is added to it as an attibute.

    Thus, when the *DataGrids* are are passed to the *ray_tracing* module this sparse grid is also traced and the \
    traced coordinates represent the centre of each pixelization pixel.

    Parameters
    -----------
    galaxies : [model.galaxy.galaxy.Galaxy]
        A list of galaxies, which may contain pixelizations and an *ImagePlanePixelization*.
    grids : data.array.grids.DataGrids
        The collection of grids (regular, sub, etc.) which the image-plane pixelization grid (referred to as pix) \
        may be added to.
    """
    if not isinstance(data_grids.regular, grids.PaddedRegularGrid):
        for galaxy in galaxies:
            if hasattr(galaxy, 'pixelization'):
                if isinstance(galaxy.pixelization, ImagePlanePixelization):

                    image_plane_pix_grid = galaxy.pixelization.image_plane_pix_grid_from_regular_grid(
                        regular_grid=data_grids.regular)
                    return data_grids.data_grids_with_pix_grid(pix_grid=image_plane_pix_grid.sparse_grid,
                           regular_to_nearest_regular_pix=image_plane_pix_grid.regular_to_sparse)

    return data_grids


class ImagePlanePixelization(object):

    def __init__(self, shape):
        """An image-plane pixelization is one where its pixel centres are computed by tracing a sparse grid of pixels \
        from the data's regular grid to other planes (e.g. the source-plane).

        The traced coordinates of this sparse grid represent each centre of a pixelization pixel.

        See *grids.SparseToRegularGrid* for details on how this grid is calculated.

        Parameters
        -----------
        shape : (float, float) or (int, int)
            The shape of the image-plane pixelizaton grid in pixels (floats are converted to integers). The grid is \
            laid over the masked data such that it spans the most outer pixels of the mask.
        """
        self.shape = (int(shape[0]), int(shape[1]))

    def image_plane_pix_grid_from_regular_grid(self, regular_grid):
        """Calculate the image-plane pixelization from a regular-grid of coordinates and its masked.

        See *grids.SparseToRegularGrid* for details on how this grid is calculated.

        Parameters
        -----------
        regular_grid : grids.RegularGrid
            The grid of (y,x) arc-second coordinates at the centre of every data value (e.g. image-pixels).
        """
        image_pixel_scale = regular_grid.mask.pixel_scale
        pixel_scales = ((regular_grid.masked_shape_arcsec[0] + image_pixel_scale) / self.shape[0],
                        (regular_grid.masked_shape_arcsec[1] + image_pixel_scale) / self.shape[1])
        return grids.SparseToRegularGrid(unmasked_sparse_grid_shape=self.shape, pixel_scales=pixel_scales,
                                         regular_grid=regular_grid)


class Pixelization(object):

    def __init__(self):
        """ Abstract base class for a pixelization, which discretizes a set of coordinates (e.g. an datas_-grid) into \
        pixels.
        """

    def mapper_from_grids_and_border(self, grids, border):
        raise NotImplementedError("pixelization_mapper_from_grids_and_borders should be overridden")

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])

    def __repr__(self):
        return "{}\n{}".format(self.__class__.__name__, str(self))


class Rectangular(Pixelization):

    def __init__(self, shape=(3, 3)):
        """A rectangular pixelization, where pixels are defined on a Cartesian and uniform grid of shape (rows, columns).

        Like an datas_, the indexing of the rectangular grid begins in the top-left corner and goes right and down.

        Parameters
        -----------
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
        """

        if shape[0] <= 2 or shape[1] <= 2:
            raise exc.PixelizationException('The rectangular pixelization must be at least dimensions 3x3')

        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]
        super(Rectangular, self).__init__()

    class Geometry(scaled_array.RectangularArrayGeometry):

        def __init__(self, shape, pixel_scales, origin, pixel_neighbors, pixel_neighbors_size):
            """The geometry of a rectangular grid

            Parameters
            -----------
            shape : (int, int)
                The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
            pixel_scales : (float, float)
                The pixel-to-arcsecond scale of a pixel in the y and x directions.
            """
            self.shape = shape
            self.pixel_scales = pixel_scales
            self.origin = origin
            self.pixel_neighbors = pixel_neighbors.astype('int')
            self.pixel_neighbors_size = pixel_neighbors_size.astype('int')

        @property
        def pixel_centres(self):
            return self.grid_1d

    def geometry_from_grid(self, grid, buffer=1e-8):
        """Determine the geometry of the rectangular grid, by alligning it with the outer-most pixels on a grid \
        plus a small buffer.

        Parameters
        -----------
        grid : [[float, float]]
            The x and y pix grid (or sub-coordinates) which are to be matched with their pixels.
        buffer : float
            The size the grid-geometry is extended beyond the most exterior grid.
        """
        y_min = np.min(grid[:, 0]) - buffer
        y_max = np.max(grid[:, 0]) + buffer
        x_min = np.min(grid[:, 1]) - buffer
        x_max = np.max(grid[:, 1]) + buffer
        pixel_scales = (float((y_max - y_min) / self.shape[0]), float((x_max - x_min) / self.shape[1]))
        origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)
        pixel_neighbors, pixel_neighbors_size = self.neighbors_from_pixelization()
        return self.Geometry(shape=self.shape, pixel_scales=pixel_scales, origin=origin,
                             pixel_neighbors=pixel_neighbors, pixel_neighbors_size=pixel_neighbors_size)

    def neighbors_from_pixelization(self):
        return pixelization_util.rectangular_neighbors_from_shape(shape=self.shape)

    def mapper_from_grids_and_border(self, grids, border):
        """Setup the pixelization mapper of this rectangular pixelization as follows:

        This first relocateds all grid-coordinates, such that any which tracer_normal beyond its border (e.g. due to high \
        levels of demagnification) are relocated to the border.

        Parameters
        ----------
        grids: mask.DataGrids
            A collection of grid describing the observed datas_'s pixel coordinates (includes an datas_ and sub grid).
        border : mask.ImagingGridBorders
            The border of the grids (defined by their datas_-plane masks).
        """

        if border is not None:
            relocated_grids = border.relocated_grids_from_grids(grids)
        else:
            relocated_grids = grids

        geometry = self.geometry_from_grid(grid=relocated_grids.sub)

        return mappers.RectangularMapper(pixels=self.pixels, grids=relocated_grids, border=border,
                                         shape=self.shape, geometry=geometry)


class Voronoi(Pixelization):

    def __init__(self):
        """Abstract base class for a Voronoi pixelization, which represents pixels as an irregular grid of Voronoi \
         pixels which can form any shape, size or tesselation.

         The traced datas_-pixels are paired to Voronoi pixels as the nearest-neighbors of the Voronoi pixel-centers.

         Parameters
         ----------
         pixels : int
             The number of pixels in the pixelization.
         """
        super(Voronoi, self).__init__()

    class Geometry(scaled_array.ArrayGeometry):

        def __init__(self, shape_arc_seconds, pixel_centres, origin, pixel_neighbors, pixel_neighbors_size):
            """The geometry of a rectangular grid

            Parameters
            -----------
            shape : (int, int)
                The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
            pixel_scales : (float, float)
                The pixel-to-arcsecond scale of a pixel in the y and x directions.
            """
            self.shape_arc_sec = shape_arc_seconds
            self.pixel_centres = pixel_centres
            self.origin = origin
            self.pixel_neighbors = pixel_neighbors.astype('int')
            self.pixel_neighbors_size = pixel_neighbors_size.astype('int')


    def geometry_from_grid(self, grid, pixel_centres, pixel_neighbors, pixel_neighbors_size, buffer=1e-8):
        """Determine the geometry of the rectangular grid, by alligning it with the outer-most pixels on a grid \
        plus a small buffer.

        Parameters
        -----------
        grid : [[float, float]]
            The x and y pix grid (or sub-coordinates) which are to be matched with their pixels.
        pixel_neighbors : [[]]
            The neighboring pix_pixels of each pix_pixel, computed via the Voronoi grid_coords. \
            (e.g. if the fifth pix_pixel neighbors pix_pixels 7, 9 and 44, pixel_neighbors[4] = [6, 8, 43])
        buffer : float
            The size the grid-geometry is extended beyond the most exterior grid.
        """
        y_min = np.min(grid[:, 0]) - buffer
        y_max = np.max(grid[:, 0]) + buffer
        x_min = np.min(grid[:, 1]) - buffer
        x_max = np.max(grid[:, 1]) + buffer
        shape_arc_seconds = (y_max - y_min, x_max - x_min)
        origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)
        return self.Geometry(shape_arc_seconds=shape_arc_seconds, pixel_centres=pixel_centres, origin=origin,
                             pixel_neighbors=pixel_neighbors, pixel_neighbors_size=pixel_neighbors_size)

    @staticmethod
    def voronoi_from_pixel_centers(pixel_centers):
        """Compute the Voronoi grid of the pixelization, using the pixel centers.

        Parameters
        ----------
        pixel_centers : ndarray
            The x and y regular_grid to derive the Voronoi grid_coords.
        """
        return scipy.spatial.Voronoi(pixel_centers, qhull_options='Qbb Qc Qx Qm')


    def neighbors_from_pixelization(self, pixels, ridge_points):
        """Compute the neighbors of every pixel as a list of the pixel index's each pixel shares a vertex with.

        The ridge points of the Voronoi grid are used to derive this.

        Parameters
        ----------
        ridge_points : scipy.spatial.Voronoi.ridge_points
            Each Voronoi-ridge (two indexes representing a pixel mapping_matrix).
        """
        return pixelization_util.voronoi_neighbors_from_pixels_and_ridge_points(pixels=pixels,
                                                                                ridge_points=np.asarray(ridge_points))


class AdaptiveMagnification(Voronoi, ImagePlanePixelization):

    def __init__(self, shape=(3, 3)):
        """A Voronoi pixelization, which traces an image-plane grid to determine the cluster-centers.

        Parameters
        ----------
        shape : (int, int)
            The shape of the regular-grid whose centres form the centres of pixelization pixels.
        """
        super(AdaptiveMagnification, self).__init__()
        ImagePlanePixelization.__init__(self=self, shape=shape)

    def mapper_from_grids_and_border(self, grids, border):
        """Setup the pixelization mapper of the cluster pixelization.

        This first relocateds all grid-coordinates, such that any which tracer_normal beyond its border (e.g. due to high \
        levels of demagnification) are relocated to the border.

        Parameters
        ----------
        grids: mask.DataGrids
            A collection of grid describing the observed datas_'s pixel coordinates (includes an datas_ and sub grid).
        border : mask.ImagingGridBorders
            The border of the grids (defined by their datas_-plane masks).
        pixel_centres : ndarray
            The center of each Voronoi pixel, computed from an traced datas_-plane grid.
        image_to_nearest_image_pix : ndarray
            The mapping of each datas_ pixel to Voronoi pixels.
        """

        if border is not None:
            relocated_grids = border.relocated_grids_from_grids(grids)
        else:
            relocated_grids = grids

        pixel_centres = relocated_grids.pix
        pixels = pixel_centres.shape[0]

        voronoi = self.voronoi_from_pixel_centers(pixel_centres)

        pixel_neighbors, pixel_neighbors_size = self.neighbors_from_pixelization(pixels=pixels,
                                                                                 ridge_points=voronoi.ridge_points)
        geometry = self.geometry_from_grid(grid=relocated_grids.sub, pixel_centres=pixel_centres,
                                           pixel_neighbors=pixel_neighbors,
                                           pixel_neighbors_size=pixel_neighbors_size)

        return mappers.VoronoiMapper(pixels=pixels, grids=relocated_grids, border=border,
                                     voronoi=voronoi, geometry=geometry)


class Amorphous(Voronoi):

    def __init__(self, pix_grid_shape):
        """
        An amorphous pixelization, which represents pixels as a set of centers where all of the \
        nearest-neighbor pix-grid (i.e. traced masked_image-pixels) are mapped to them.

        For this pixelization, a set of cluster-pixels (defined in the masked_image-plane as a cluster uniform grid of \
        masked_image-pixels) are used to determine a set of pix-plane grid. These grid are then fed into a \
        weighted k-means clustering algorithm, such that the pixel centers adapt to the unlensed pix \
        surface-brightness profile.

        Parameters
        ----------
        pix_grid_shape : (int, int)
            The shape of the regular-grid whose centres form the centres of pixelization pixels.
        """
        super(Amorphous, self).__init__()

    def kmeans_cluster(self, pixels, cluster_grid):
        """Perform k-means clustering on the cluster_grid to compute the k-means clusters which represent \
        pixels.

        Parameters
        ----------
        cluster_grid : ndarray
            The x and y cluster-grid which are used to derive the k-means pixelization.
        """
        kmeans = sklearn.cluster.KMeans(pixels)
        km = kmeans.fit(cluster_grid)
        return km.cluster_centers_, km.labels_