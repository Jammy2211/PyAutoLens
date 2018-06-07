import numpy as np
import sklearn.cluster
import scipy.spatial


class VoronoiPixelization(object):

    def __init__(self, number_clusters, regularization_coefficient=1.0):
        self.number_clusters = number_clusters
        self.regularizatioon_coefficient = regularization_coefficient

    def compute_mapping_matrix(self, sub_coordinates, cluster_coordinates, image_to_cluster):
        kmeans = self.kmeans_cluster(cluster_coordinates)
        voronoi = compute_voronoi_grid(cluster_coordinates)
        sub_to_source = compute_sub_to_source(sub_coordinates, kmeans.cluster_centers_, voronoi.neighbors,
                                              image_to_cluster, kmeans.labels_)

    # return create_mapping_matrix()

    def kmeans_cluster(self, coordinates):
        """Perform k-means clustering on a set of image_grid to compute the set of k-means clustering that group the data.

        This is used to compute the source-pixel centers of the *AmorphousPixelization*

        Parameters
        ----------
        coordinates : ndarray
            The x and y image_grid to be clustered.
        n_clusters : int
            The number of clusters to cluster the data with.
        """
        kmeans = sklearn.cluster.KMeans(self.number_clusters)
        return kmeans.fit(coordinates)


def compute_voronoi_grid(coordinates):
    """Setup a Voronoi grid_coords for a given set of image_grid, as well as a list of each Voronoi cell's neighboring \
     Voronoi cells.

    This is used to compute the Voronoi grid_coords of the source-pixel centers of the *AmorphousPixelization*.

    Parameters
    ----------
    coordinates : ndarray
        The x and y image_grid to derive the Voronoi grid_coords.
    """
    voronoi = scipy.spatial.Voronoi(coordinates, qhull_options='Qbb Qc Qx Qm')

    voronoi.neighbors = [[] for _ in range(len(coordinates))]

    for pair in reversed(voronoi.ridge_points):
        voronoi.neighbors[pair[0]].append(pair[1])
        voronoi.neighbors[pair[1]].append(pair[0])

    voronoi.neighbors_total = list(map(lambda x: len(x), voronoi.neighbors))
    return voronoi


def compute_sub_to_source(sub_coordinates, source_centers, source_neighbors, image_to_sparse, sparse_to_source):
    """ Match a set of sub grid coordinates to their closest source-pixel, using the source-pixel centers (x,y).

        This method uses a sparsely sampled grid of coordinates, known mapping to the image pixels and source pixels.
        Thus, the sparse_grid grid_coords of sub_grid must have had a source \
        pixelization derived (e.g. using the KMeans class) and the neighbors of each source-pixel must be known \
        (e.g. using the Voronoi class). Both must have been performed prior to this function call.

        In a realistic lens pixelization, the sparse_grid image_grid will correspond to the center of each image_grid pixel \
        (traced to the source-plane) or a sparser grid_coords of image_grid-image_to_pixel. The sub_grid will be the sub_grid \
        image_grid-image_to_pixel (again, traced to the source-plane). A benefit of this is the source-pixelization (e.g. using \
        KMeans) will be dervied using significantly fewer sub_grid, offering run-time speedup.

        In the routine below, some variables and function names refer to a 'sparse_source_'. This term describes a \
        source-pixel that we have paired to a sub_coordinate using the sparse_grid grid_coords of image_grid image_to_pixel. Thus, it may not \
        actually be that sub_coordinate's closest source-pixel (the routine will eventually determine this).

        Parameters
        ----------
        sub_coordinates : [(float, float)]
            The x and y sub_grid to be matched to the source_pixel centers.
        image_total : int
            The total number of image_grid image_to_pixel in the image_grid.
        sub_total : int
            The total number of sub_grid image_to_pixel in the image_grid sub_grid-grid_coords.
        source_centers: [(float, float)]
            The source_pixel centers the sub_grid are matched with.
        source_neighbors : [[]]
            The neighboring source_pixels of each source_pixel, computed via the Voronoi grid_coords (e.g. if the fifth source_pixel \
            neighbors source_pixels 7, 9 and 44, source_neighbors[4] = [6, 8, 43])
        sub_to_sparse : [int]
            The index in sub_grid each sparse_grid sub_coordinate is closest too (e.g. if the fifth sparse_grid sub_coordinate \
            is closest to the 3rd sub_coordinate in sub_grid, sub_to_sparse[4] = 2).
        sparse_to_source : [int]
            The index in source_pixel_centers each sparse_grid sub_coordinate closest too (e.g. if the fifth sparse_grid sub_coordinate \
            is closest to the 3rd source_pixel in source_pixel_centers, sparse_to_source[4] = 2).

        Returns
        ----------
        sub_to_source : [int, int]
            The index in source_pixel_centers each image_grid and sub_grid-image_coordinate is matched with. (e.g. if the fifth
            sub_coordinate of the third image_grid pixel is closest to the 3rd source-pixel in source_pixel_centers,
            sub_to_source[2,4] = 2).

     """

    image_pixels = sub_coordinates.shape[0]
    sub_pixels = sub_coordinates.shape[0] * sub_coordinates.shape[1]

    sub_to_source = np.zeros((image_pixels, sub_pixels))

    for image_index in range(len(sub_coordinates)):
        sub_index = 0
        for sub_coordinate in sub_coordinates[image_index]:

            nearest_sparse = image_to_sparse[image_index]
            nearest_sparse_source = sparse_to_source[nearest_sparse]

            while True:

                sub_to_sparse_source_distance = compute_sub_to_nearest_sparse_source(source_centers,
                                                                                     sub_coordinate,
                                                                                     nearest_sparse_source)

                neighboring_source_index, sub_to_neighboring_source_distance = \
                    compute_nearest_neighboring_source_and_distance(sub_coordinate, source_centers,
                                                                    source_neighbors[nearest_sparse_source])

                if sub_to_sparse_source_distance < sub_to_neighboring_source_distance:
                    break
                else:
                    nearest_sparse_source = neighboring_source_index

            # If this pixel is closest to the original pixel, it has been paired successfully with its nearest neighbor.
            sub_to_source[image_index, sub_index] = nearest_sparse_source
            sub_index += 1

    return sub_to_source


def compute_sub_to_nearest_sparse_source(source_centers, sub_coordinate, source_pixel):
    nearest_sparse_source_pixel_center = source_centers[source_pixel]
    return compute_squared_separation(sub_coordinate, nearest_sparse_source_pixel_center)


def compute_nearest_neighboring_source_and_distance(sub_coordinate, source_centers, source_neighbors):
    """For a given source_pixel, we look over all its adjacent neighbors and find the neighbor whose distance is closest to
    our input coordinaates.

        Parameters
        ----------
        sub_coordinate : (float, float)
            The x and y coordinate to be matched with the neighboring set of source_pixels.
        source_centers: [(float, float)
            The source_pixel centers the image_grid are matched with.
        source_neighbors : list
            The neighboring source_pixels of the sparse_grid source_pixel the coordinate is currently matched with

        Returns
        ----------
        source_pixel_neighbor_index : int
            The index in source_pixel_centers of the closest source_pixel neighbor.
        source_pixel_neighbor_separation : float
            The separation between the input coordinate and closest source_pixel neighbor

    """

    separation_from_neighbor = list(map(lambda neighbors:
                                        compute_squared_separation(sub_coordinate, source_centers[neighbors]),
                                        source_neighbors))

    closest_separation_index = min(range(len(separation_from_neighbor)), key=separation_from_neighbor.__getitem__)

    return source_neighbors[closest_separation_index], separation_from_neighbor[closest_separation_index]


def compute_squared_separation(coordinate1, coordinate2):
    """Computes the squared separation of two image_grid (no square root for efficiency)"""
    return (coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2


# TODO : Remove image_pixel_total, sub_grid_size, sub_to_image from input, as can be derived from sub_to_source
# TODO : Make a method in Pixelization class, thus also removing source_pixel_total.

def create_mapping_matrix(source_pixel_total, image_pixel_total, sub_grid_size, sub_to_source, sub_to_image):
    """
    Set up a new mapping matrix, which describes the fractional unit surface brightness counts between each
    image_grid-pixel and source pixel pair.

    The mapping matrix is the matrix denoted 'f_ij' in Warren & Dye 2003, Nightingale & Dye 2015 and Nightingale,
    Dye & Massey 2018.

    It is a matrix of pixel_dimensions [source_pixels x image_pixels], wherein a non-zero entry represents an
    image_grid-pixel to source-pixel mapping. For example, if image_grid-pixel 4 maps to source-pixel 2, then element
    (2,4) of the mapping matrix will = 1.

    The mapping matrix supports sub_grid-gridding.  Here, each image_grid-pixel in the observed image_grid is divided
    into a finer sub_grid-grid_coords. For example, if grid_size_sub = 4, each image_grid-pixel is split into a 4 x 4
    sub_grid-grid_coords, giving a total of 16 sub_grid image_grid-image_to_pixel. All 16 sub_grid
    image_grid-image_to_pixel are individually mapped to the source-plane and each is paired with a source-pixel.

    The entries in the mapping matrix now become fractional values representing the number of sub_grid
    image_grid-image_to_pixel which map to each source-pixel. For example if 3 sub_grid image_grid-image_to_pixel within
    image_grid-pixel 4 map to source-pixel 2, and the grid_size_sub=2, then element (2,4) of the mapping matrix
    will = 3.0 * (1/grid_size_sub**2) = 3/16 = 0.1875.

    Parameters
    ----------
    source_pixel_total : int
        The number of source-image_to_pixel in the source-plane (and first dimension of the mapping matrix)
    image_pixel_total : int
        The number of image_grid-image_to_pixel in the masked observed image_grid (and second dimension of the mapping
        matrix)
    sub_grid_size : int
        The size of sub_grid-gridding used on the observed image_grid.
    sub_to_source : [int]
        The index of the source_pixel each image_grid sub_grid-pixel is mapped too (e.g. if the fifth sub_grid
        image_grid pixel is mapped to the 3rd source_pixel in the source plane, sub_to_source[4] = 2).
    sub_to_image : [int]
        The index of the image_grid-pixel each image_grid sub_grid-pixel belongs too (e.g. if the fifth sub_grid
        image_grid pixel is within the 3rd image_grid-pixel in the observed image_grid, sub_to_image[4] = 2).
    """

    total_sub_pixels = image_pixel_total * sub_grid_size ** 2
    sub_grid_fraction = (1.0 / sub_grid_size) ** 2

    f = [{} for _ in range(source_pixel_total)]

    for i in range(total_sub_pixels):
        if sub_to_image[i] in f[sub_to_source[i]]:
            f[sub_to_source[i]][sub_to_image[i]] += sub_grid_fraction
        else:
            f[sub_to_source[i]][sub_to_image[i]] = sub_grid_fraction

    return f


def setup_regularization_matrix_via_pixel_pairs(dimension, regularization_weights, no_pairs, pixel_pairs):
    """
    Setup a new regularization matrix, bypassing matrix multiplication by exploiting a list of pixel-pairs.

    Parameters
    ----------
    dimension : int
        The pixel_dimensions of the square regularization matrix
    regularization_weights : list(float)
        The regularization weight of each source-pixel
    no_pairs : list(int)
        The number of pairs each source-plane pixel shares with its other image_to_pixel
    pixel_pairs : list(float, float)
        A list of all pixel-pairs in the source-plane, as computed by the Voronoi gridding routine.
    """

    matrix = np.zeros(shape=(dimension, dimension))

    reg_weight = regularization_weights ** 2

    for i in range(dimension):
        matrix[i][i] += no_pairs[i] * reg_weight[i]

    for j in range(len(pixel_pairs)):
        matrix[pixel_pairs[j, 0], pixel_pairs[j, 0]] += reg_weight[pixel_pairs[j, 1]]
        matrix[pixel_pairs[j, 1], pixel_pairs[j, 1]] += reg_weight[pixel_pairs[j, 0]]
        matrix[pixel_pairs[j, 0], pixel_pairs[j, 1]] -= reg_weight[pixel_pairs[j, 0]]
        matrix[pixel_pairs[j, 1], pixel_pairs[j, 0]] -= reg_weight[pixel_pairs[j, 0]]
        matrix[pixel_pairs[j, 0], pixel_pairs[j, 1]] -= reg_weight[pixel_pairs[j, 1]]
        matrix[pixel_pairs[j, 1], pixel_pairs[j, 0]] -= reg_weight[pixel_pairs[j, 1]]

    return matrix
