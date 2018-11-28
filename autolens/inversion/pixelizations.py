import numpy as np
import scipy.spatial
import sklearn.cluster

from autolens import exc
from autolens.imaging import mask
from autolens.imaging import scaled_array
from autolens.inversion.util import pixelization_util
from autolens.inversion import mappers


class ImagePlanePixelizationGrid(scaled_array.ArrayGeometry):

    def __init__(self, pix_grid_shape, pixel_scales, image_grid):
        """Abstract class which handles the uniform image-grid whose pixel centers are used to form an adaptive grid's \
        pixelization pixel-centers.

        This is performed by over-laying a mask over the image-grid, such that only pixels within masked pixels are \
        included in the pixelization.

        Parameters
        ----------
        pix_grid_shape : (int, int)
            The shape of the image-grid whose centres form the centres of pixelization pixels.
        pixel_scales : (float, float)
            The pixel-to-arcsecond scale of a pixel in the y and x directions.
        """

        self.shape = pix_grid_shape
        self.origin = (0.0, 0.0)
        self.total_pixels = int(self.shape[0] * self.shape[1])
        self.pixel_scales = pixel_scales

        self.image_grid = image_grid
        self.full_pix_grid = self.grid_1d
        self.full_pix_grid_pixel_centres = image_grid.mask.grid_arc_seconds_to_grid_pixel_centres(self.full_pix_grid)

    @property
    def total_pix_pixels(self):
        return pixelization_util.total_pix_pixels_from_mask(mask=self.image_grid.mask,
                                     full_pix_grid_pixel_centres=self.full_pix_grid_pixel_centres)

    @property
    def full_pix_to_pix(self):
        """The 1D index mappings between the unmasked pixelization-grid and masked pixelization grid."""
        return pixelization_util.full_pix_to_pix_from_mask_and_pixel_centres(mask=self.image_grid.mask,
                                         full_pix_grid_pixel_centres=self.full_pix_grid_pixel_centres).astype('int')

    @property
    def pix_to_full_pix(self):
        """The 1D index mappings between the masked pixelization-grid and unmasked pixelization grid."""
        return pixelization_util.pix_to_full_pix_from_mask_and_pixel_centres(total_pix_pixels=self.total_pix_pixels,
              mask=self.image_grid.mask, full_pix_grid_pixel_centres=self.full_pix_grid_pixel_centres).astype('int')

    @property
    def image_to_full_pix(self):
        return self.grid_arc_seconds_to_grid_pixel_indexes(grid_arc_seconds=self.image_grid)

    @property
    def image_to_pix(self):
        return pixelization_util.image_to_pix_from_pix_mappings(image_to_full_pix=self.image_to_full_pix,
                                                                full_pix_to_pix=self.full_pix_to_pix).astype('int')

    @property
    def pix_grid(self):
        return pixelization_util.pix_grid_from_full_pix_grid(full_pix_grid=self.full_pix_grid,
                                                                      pix_to_full_pix=self.pix_to_full_pix)


class ImagePlanePixelization(object):

    def __init__(self, pix_grid_shape):

        self.pix_grid_shape = pix_grid_shape

    def image_plane_pix_grid_from_image_grid(self, image_grid):
        image_pixel_scale = image_grid.mask.pixel_scale
        pixel_scales = ((image_grid.masked_shape_arcsec[0] + image_pixel_scale) / self.pix_grid_shape[0],
                        (image_grid.masked_shape_arcsec[1] + image_pixel_scale) / self.pix_grid_shape[1])
        return ImagePlanePixelizationGrid(pix_grid_shape=self.pix_grid_shape, pixel_scales=pixel_scales,
                                          image_grid=image_grid)


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

    class Geometry(scaled_array.ArrayGeometry):

        def __init__(self, shape, pixel_scales, origin):
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
        return self.Geometry(shape=self.shape, pixel_scales=pixel_scales, origin=origin)

    def neighbors_from_pixelization(self):
        """Compute the neighbors of every pixel as a list of the pixel index's each pixel shares a vertex with.

        The uniformity of the rectangular grid's geometry is used to compute this.
        """

        def compute_corner_neighbors(pixel_neighbors):

            pixel_neighbors[0] = [1, self.shape[1]]
            pixel_neighbors[self.shape[1] - 1] = [self.shape[1] - 2, self.shape[1] + self.shape[1] - 1]
            pixel_neighbors[self.pixels - self.shape[1]] = [self.pixels - self.shape[1] * 2,
                                                            self.pixels - self.shape[1] + 1]
            pixel_neighbors[self.pixels - 1] = [self.pixels - self.shape[1] - 1, self.pixels - 2]

            return pixel_neighbors

        def compute_top_edge_neighbors(pixel_neighbors):

            for pix in range(1, self.shape[1] - 1):
                pixel_index = pix
                pixel_neighbors[pixel_index] = [pixel_index - 1, pixel_index + 1, pixel_index + self.shape[1]]

            return pixel_neighbors

        def compute_left_edge_neighbors(pixel_neighbors):

            for pix in range(1, self.shape[0] - 1):
                pixel_index = pix * self.shape[1]
                pixel_neighbors[pixel_index] = [pixel_index - self.shape[1], pixel_index + 1,
                                                pixel_index + self.shape[1]]

            return pixel_neighbors

        def compute_right_edge_neighbors(pixel_neighbors):

            for pix in range(1, self.shape[0] - 1):
                pixel_index = pix * self.shape[1] + self.shape[1] - 1
                pixel_neighbors[pixel_index] = [pixel_index - self.shape[1], pixel_index - 1,
                                                pixel_index + self.shape[1]]

            return pixel_neighbors

        def compute_bottom_edge_neighbors(pixel_neighbors):

            for pix in range(1, self.shape[1] - 1):
                pixel_index = self.pixels - pix - 1
                pixel_neighbors[pixel_index] = [pixel_index - self.shape[1], pixel_index - 1, pixel_index + 1]

            return pixel_neighbors

        def compute_central_neighbors(pixel_neighbors):

            for x in range(1, self.shape[0] - 1):
                for y in range(1, self.shape[1] - 1):
                    pixel_index = x * self.shape[1] + y
                    pixel_neighbors[pixel_index] = [pixel_index - self.shape[1], pixel_index - 1, pixel_index + 1,
                                                    pixel_index + self.shape[1]]

            return pixel_neighbors

        pixel_neighbors = [[] for _ in range(self.pixels)]

        pixel_neighbors = compute_corner_neighbors(pixel_neighbors)
        pixel_neighbors = compute_top_edge_neighbors(pixel_neighbors)
        pixel_neighbors = compute_left_edge_neighbors(pixel_neighbors)
        pixel_neighbors = compute_right_edge_neighbors(pixel_neighbors)
        pixel_neighbors = compute_bottom_edge_neighbors(pixel_neighbors)
        pixel_neighbors = compute_central_neighbors(pixel_neighbors)

        return pixel_neighbors

    def mapper_from_grids_and_border(self, grids, border):
        """Setup the pixelization mapper of this rectangular pixelization as follows:

        This first relocateds all grid-coordinates, such that any which tracer_normal beyond its border (e.g. due to high \
        levels of demagnification) are relocated to the border.

        Parameters
        ----------
        grids: mask.ImagingGrids
            A collection of grid describing the observed datas_'s pixel coordinates (includes an datas_ and sub grid).
        border : mask.ImagingGridBorders
            The border of the grids (defined by their datas_-plane masks).
        """

        if border is not None:
            relocated_grids = border.relocated_grids_from_grids(grids)
        else:
            relocated_grids = grids

        geometry = self.geometry_from_grid(relocated_grids.sub)
        pixel_neighbors = self.neighbors_from_pixelization()
        return mappers.RectangularMapper(pixels=self.pixels, grids=relocated_grids, border=border,
                                         pixel_neighbors=pixel_neighbors, shape=self.shape, geometry=geometry)


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

    @staticmethod
    def voronoi_from_pixel_centers(pixel_centers):
        """Compute the Voronoi grid of the pixelization, using the pixel centers.

        Parameters
        ----------
        pixel_centers : ndarray
            The x and y image_grid to derive the Voronoi grid_coords.
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
        pixel_neighbors = [[] for _ in range(pixels)]

        for pair in reversed(ridge_points):
            pixel_neighbors[pair[0]].append(pair[1])
            pixel_neighbors[pair[1]].append(pair[0])

        return pixel_neighbors


class AdaptiveMagnification(Voronoi, ImagePlanePixelization):

    def __init__(self, pix_grid_shape):
        """A cluster pixelization, which represents pixels as a Voronoi grid (see Voronoi). For this pixelization, the \
        pixel-centers are derived from a sparse-grid in the observed datas_ which are lensed to form the pixel centers.

        Parameters
        ----------
        pix_grid_shape : (int, int)
            The shape of the image-grid whose centres form the centres of pixelization pixels.
        """
        ImagePlanePixelization.__init__(self=self, pix_grid_shape=pix_grid_shape)
        super(AdaptiveMagnification, self).__init__()

    def mapper_from_grids_and_border(self, grids, border, image_to_nearest_image_pix):
        """Setup the pixelization mapper of the cluster pixelization.

        This first relocateds all grid-coordinates, such that any which tracer_normal beyond its border (e.g. due to high \
        levels of demagnification) are relocated to the border.

        Parameters
        ----------
        grids: mask.ImagingGrids
            A collection of grid describing the observed datas_'s pixel coordinates (includes an datas_ and sub grid).
        border : mask.ImagingGridBorders
            The border of the grids (defined by their datas_-plane masks).
        pixel_centers : ndarray
            The center of each Voronoi pixel, computed from an traced datas_-plane grid.
        image_to_nearest_image_pix : ndarray
            The mapping of each datas_ pixel to Voronoi pixels.
        """

        pixel_centers = grids.pix
        pixels = pixel_centers.shape[0]

        if border is not None:
            relocated_grids = border.relocated_grids_from_grids(grids)
        else:
            relocated_grids = grids

        voronoi = self.voronoi_from_pixel_centers(pixel_centers)
        pixel_neighbors = self.neighbors_from_pixelization(pixels=pixels, ridge_points=voronoi.ridge_points)

        return mappers.VoronoiMapper(pixels=pixels, grids=relocated_grids, border=border,
                                     pixel_neighbors=pixel_neighbors,
                                     pixel_centers=pixel_centers, voronoi=voronoi,
                                     image_to_nearest_image_pix=image_to_nearest_image_pix)


class Amorphous(Voronoi):

    def __init__(self, image_grid_shape):
        """
        An amorphous pixelization, which represents pixels as a set of centers where all of the \
        nearest-neighbor pix-grid (i.e. traced masked_image-pixels) are mapped to them.

        For this pixelization, a set of cluster-pixels (defined in the masked_image-plane as a cluster uniform grid of \
        masked_image-pixels) are used to determine a set of pix-plane grid. These grid are then fed into a \
        weighted k-means clustering algorithm, such that the pixel centers adapt to the unlensed pix \
        surface-brightness profile.

        Parameters
        ----------
        image_grid_shape : (int, int)
            The shape of the image-grid whose centres form the centres of pixelization pixels.
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