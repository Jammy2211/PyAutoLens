import numpy as np
import scipy.spatial
import sklearn.cluster

from autolens import exc
from autolens.imaging import mask
from autolens.imaging import scaled_array
from autolens.inversion import mappers


class Pixelization(object):

    def __init__(self, pixels=100):
        """ Abstract base class for a pixelization, which discretizes a set of coordinates (e.g. an datas_-grid) into \
        pixels.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        """
        self.pixels = pixels

    def mapper_from_grids_and_borders(self, grids, borders):
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

        super(Rectangular, self).__init__(self.shape[0] * self.shape[1])

    class Geometry(scaled_array.ArrayGeometry):

        def __init__(self, shape, pixel_scales):
            """The geometry of a rectangular grid

            Parameters
            -----------
            y_min : float
                Minimum arcsecond y value of the pixelization (e.g. the top-edge).
            y_max : float
                Maximum arcsecond y value of the pixelization (e.g. the bottom-edge).
            x_min : float
                Minimum arcsecond x value of the pixelization (e.g. the left-edge).
            x_max : float
                Maximum arcsecond x value of the pixelization (e.g. the right-edge).
            pixel_scales : (float, float)
                The pixel-to-arcsecond scale of a pixel in the y and x directions.
            """
            self.shape = shape
            self.pixel_scales = pixel_scales

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
        pixel_scales = ((y_max - y_min) / self.shape[0], (x_max - x_min) / self.shape[1])

        return self.Geometry(shape=self.shape, pixel_scales=pixel_scales)

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

    def mapper_from_grids(self, grids):
        """Setup the pixelization mapper of this rectangular pixelization as follows:

        This first relocateds all grid-coordinates, such that any which tracer_normal beyond its border (e.g. due to high \
        levels of demagnification) are relocated to the border.

        Parameters
        ----------
        grids: mask.ImagingGrids
            A collection of grid describing the observed datas_'s pixel coordinates (includes an datas_ and sub grid).
        borders : mask.ImagingGridBorders
            The borders of the grids (defined by their datas_-plane masks).
        """
        geometry = self.geometry_from_grid(grids.sub)
        pixel_neighbors = self.neighbors_from_pixelization()
        return mappers.RectangularMapper(pixels=self.pixels, grids=grids, pixel_neighbors=pixel_neighbors,
                                         shape=self.shape, geometry=geometry)

    def mapper_from_grids_and_borders(self, grids, borders):
        """Setup the pixelization mapper of this rectangular pixelization as follows:

        This first relocateds all grid-coordinates, such that any which tracer_normal beyond its border (e.g. due to high \
        levels of demagnification) are relocated to the border.

        Parameters
        ----------
        grids: mask.ImagingGrids
            A collection of grid describing the observed datas_'s pixel coordinates (includes an datas_ and sub grid).
        borders : mask.ImagingGridBorders
            The borders of the grids (defined by their datas_-plane masks).
        """
        try:
            relocated_grids = borders.relocated_grids_from_grids(grids)
        except ValueError:
            relocated_grids = grids

        geometry = self.geometry_from_grid(relocated_grids.sub)
        pixel_neighbors = self.neighbors_from_pixelization()
        return mappers.RectangularMapper(pixels=self.pixels, grids=relocated_grids, pixel_neighbors=pixel_neighbors,
                                         shape=self.shape, geometry=geometry)


class Voronoi(Pixelization):

    def __init__(self, pixels=100):
        """Abstract base class for a Voronoi pixelization, which represents pixels as an irregular grid of Voronoi \
         pixels which can form any shape, size or tesselation.

         The traced datas_-pixels are paired to Voronoi pixels as the nearest-neighbors of the Voronoi pixel-centers.

         Parameters
         ----------
         pixels : int
             The number of pixels in the pixelization.
         """
        super(Voronoi, self).__init__(pixels)

    @staticmethod
    def voronoi_from_pixel_centers(pixel_centers):
        """Compute the Voronoi grid of the pixelization, using the pixel centers.

        Parameters
        ----------
        pixel_centers : ndarray
            The x and y image_grid to derive the Voronoi grid_coords.
        """
        return scipy.spatial.Voronoi(pixel_centers, qhull_options='Qbb Qc Qx Qm')

    def neighbors_from_pixelization(self, ridge_points):
        """Compute the neighbors of every pixel as a list of the pixel index's each pixel shares a vertex with.

        The ridge points of the Voronoi grid are used to derive this.

        Parameters
        ----------
        ridge_points : scipy.spatial.Voronoi.ridge_points
            Each Voronoi-ridge (two indexes representing a pixel mapping_matrix).
        """
        pixel_neighbors = [[] for _ in range(self.pixels)]

        for pair in reversed(ridge_points):
            pixel_neighbors[pair[0]].append(pair[1])
            pixel_neighbors[pair[1]].append(pair[0])

        return pixel_neighbors


class Cluster(Voronoi):

    def __init__(self, pixels):
        """A cluster pixelization, which represents pixels as a Voronoi grid (see Voronoi). For this pixelization, the \
        pixel-centers are derived from a sparse-grid in the observed datas_ which are lensed to form the pixel centers.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        """
        super(Cluster, self).__init__(pixels)

    def mapper_from_grids(self, grids, pixel_centers, image_to_voronoi):
        """Setup the pixelization mapper of the cluster pixelization.

        This first relocateds all grid-coordinates, such that any which tracer_normal beyond its border (e.g. due to high \
        levels of demagnification) are relocated to the border.

        Parameters
        ----------
        grids: mask.ImagingGrids
            A collection of grid describing the observed datas_'s pixel coordinates (includes an datas_ and sub grid).
        borders : mask.ImagingGridBorders
            The borders of the grids (defined by their datas_-plane masks).
        pixel_centers : ndarray
            The center of each Voronoi pixel, computed from an traced datas_-plane grid.
        image_to_voronoi : ndarray
            The mapping of each datas_ pixel to Voronoi pixels.
        """

        voronoi_to_pixelization = np.arange(0, self.pixels)
        voronoi = self.voronoi_from_pixel_centers(pixel_centers)
        pixel_neighbors = self.neighbors_from_pixelization(voronoi.ridge_points)

        return mappers.VoronoiMapper(pixels=self.pixels, grids=grids,
                                     pixel_neighbors=pixel_neighbors,
                                     pixel_centers=pixel_centers, voronoi=voronoi,
                                     voronoi_to_pixelization=voronoi_to_pixelization,
                                     image_to_voronoi=image_to_voronoi)

    def mapper_from_grids_and_borders(self, grids, borders, pixel_centers, image_to_voronoi):
        """Setup the pixelization mapper of the cluster pixelization.

        This first relocateds all grid-coordinates, such that any which tracer_normal beyond its border (e.g. due to high \
        levels of demagnification) are relocated to the border.

        Parameters
        ----------
        grids: mask.ImagingGrids
            A collection of grid describing the observed datas_'s pixel coordinates (includes an datas_ and sub grid).
        borders : mask.ImagingGridBorders
            The borders of the grids (defined by their datas_-plane masks).
        pixel_centers : ndarray
            The center of each Voronoi pixel, computed from an traced datas_-plane grid.
        image_to_voronoi : ndarray
            The mapping of each datas_ pixel to Voronoi pixels.
        """

        relocated_grids = borders.relocated_grids_from_grids(grids)
        voronoi_to_pixelization = np.arange(0, self.pixels)
        voronoi = self.voronoi_from_pixel_centers(pixel_centers)
        pixel_neighbors = self.neighbors_from_pixelization(voronoi.ridge_points)

        return mappers.VoronoiMapper(pixels=self.pixels, grids=relocated_grids,
                                     pixel_neighbors=pixel_neighbors,
                                     pixel_centers=pixel_centers, voronoi=voronoi,
                                     voronoi_to_pixelization=voronoi_to_pixelization,
                                     image_to_voronoi=image_to_voronoi)


class Amorphous(Voronoi):

    def __init__(self, pixels):
        """
        An amorphous pixelization, which represents pixels as a set of centers where all of the \
        nearest-neighbor pix-grid (i.e. traced masked_image-pixels) are mapped to them.

        For this pixelization, a set of cluster-pixels (defined in the masked_image-plane as a cluster uniform grid of \
        masked_image-pixels) are used to determine a set of pix-plane grid. These grid are then fed into a \
        weighted k-means clustering algorithm, such that the pixel centers adapt to the unlensed pix \
        surface-brightness profile.

        Parameters
        ----------
        pixels : int
            The number of pixels in the pixelization.
        regularization_coefficients : (float,)
            The regularization_matrix coefficients used to smooth the pix reconstructed_image.
        """
        super(Amorphous, self).__init__(pixels)

    def kmeans_cluster(self, cluster_grid):
        """Perform k-means clustering on the cluster_grid to compute the k-means clusters which represent \
        pixels.

        Parameters
        ----------
        cluster_grid : ndarray
            The x and y cluster-grid which are used to derive the k-means pixelization.
        """
        kmeans = sklearn.cluster.KMeans(self.pixels)
        km = kmeans.fit(cluster_grid)
        return km.cluster_centers_, km.labels_


class PixelizationGrid(object):

    def __init__(self, shape=(3, 3)):
        self.shape = shape

    def coordinate_grid_within_annulus(self, inner_radius, outer_radius, centre=(0., 0.)):

        y_pixel_scale = 2.0 * outer_radius / self.shape[0]
        x_pixel_scale = 2.0 * outer_radius / self.shape[1]

        central_pixel = float(self.shape[0] - 1) / 2, float(self.shape[1] - 1) / 2

        pixel_count = 0

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):

                y_arcsec = ((y - central_pixel[1]) * y_pixel_scale) - centre[0]
                x_arcsec = ((x - central_pixel[0]) * x_pixel_scale) - centre[1]
                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                if radius_arcsec < outer_radius or radius_arcsec > inner_radius:
                    pixel_count += 1

        coordinates_array = np.zeros((pixel_count, 2))

        pixel_count = 0

        for y in range(self.shape[0]):
            for x in range(self.shape[1]):

                y_arcsec = -((y - central_pixel[0]) * y_pixel_scale) - centre[0]
                x_arcsec = ((x - central_pixel[1]) * x_pixel_scale) - centre[1]
                radius_arcsec = np.sqrt(x_arcsec ** 2 + y_arcsec ** 2)

                if radius_arcsec < outer_radius or radius_arcsec > inner_radius:
                    coordinates_array[pixel_count, 0] = y_arcsec
                    coordinates_array[pixel_count, 1] = x_arcsec

                    pixel_count += 1

        return coordinates_array
