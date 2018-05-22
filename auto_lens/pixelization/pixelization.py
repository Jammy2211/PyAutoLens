import numpy as np
import sklearn.cluster
import scipy.spatial


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
        The number of pairs each source-plane pixel shares with its other data_to_pixels
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

def kmeans_cluster(coordinates, n_clusters):
    """Perform k-means clustering on a set of image_grid to compute the set of k-means clustering that group the data.

    This is used to compute the source-pixel centers of the *AmorphousPixelization*

    Parameters
    ----------
    coordinates : ndarray
        The x and y image_grid to be clustered.
    n_clusters : int
        The number of clusters to cluster the data with.
    """
    kmeans = sklearn.cluster.KMeans(n_clusters)
    return kmeans.fit(coordinates)

def setup_voronoi(coordinates):
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

def sub_coordinates_to_source_pixels_via_nearest_neighbour(sub_coordinates, image_total, sub_total, source_centers):
    """ Match a set of sub_grid image_grid-pixel image_grid to their closest source-data_to_pixels, using the source-pixel centers (x,y).

        This method uses a nearest neighbour search between every sub_image-pixel coordinate and set of source-pixel \
        centers, thus it is slow when the number of sub_grid image_grid-pixel image_grid or source-data_to_pixels is large. However, it
        is probably the fastest routine for low numbers of sub_grid image_grid-data_to_pixels and source-data_to_pixels.

        Parameters
        ----------
        sub_coordinates : [(float, float)]
            The x and y sub_grid image_grid-pixel image_grid to be matched to the source-pixel centers.
        image_total : int
            The total number of image_grid data_to_pixels in the image_grid.
        sub_total : int
            The total number of sub_grid data_to_pixels in the image_grid sub_grid-grid_coords.
        source_centers: [(float, float)
            The source-data_to_pixels centers the sub_grid image_grid-pixel image_grid are matched with.

        Returns
        ----------
        image_sub_to_source : [int, int]
            The index in source_pixel_centers each image_grid and sub_grid-image_coordinate is matched with. (e.g. if the fifth
            sub_coordinate of the third image_grid pixel is closest to the 3rd source-pixel in source_pixel_centers,
            image_sub_to_source[2,4] = 2).

     """

    image_sub_to_source = np.zeros((image_total, sub_total))

    for image_index in range(len(sub_coordinates)):
        sub_index = 0
        for sub_coordinate in sub_coordinates[image_index]:
            distances = list(map(lambda centers: compute_squared_separation(sub_coordinate, centers), source_centers))

            image_sub_to_source[image_index, sub_index] = (np.argmin(distances))
            sub_index += 1

    return image_sub_to_source

def sub_coordinates_to_source_pixels_via_sparse_pairs(sub_coordinates, image_total, sub_total, source_centers,
                                                      source_neighbors, image_to_sparse, sparse_to_source):
    """ Match a set of sub_grid image_grid-pixel image_grid to their closest source-pixel, using the source-pixel centers (x,y).

        This method uses a sparsely sampled grid_coords of sub_grid image_grid-pixel image_grid with known image_grid-pixel to source-pixel \
        pairings and the source-data_to_pixels neighbors to speed up the function. This is optimal when the number of sub_grid \
        image_grid-data_to_pixels or source-data_to_pixels is large. Thus, the sparse_grid grid_coords of sub_grid must have had a source \
        pixelization derived (e.g. using the KMeans class) and the neighbors of each source-pixel must be known \
        (e.g. using the Voronoi class). Both must have been performed prior to this function call.

        In a realistic lens pixelization, the sparse_grid image_grid will correspond to the center of each image_grid pixel \
        (traced to the source-plane) or a sparser grid_coords of image_grid-data_to_pixels. The sub_grid will be the sub_grid \
        image_grid-data_to_pixels (again, traced to the source-plane). A benefit of this is the source-pixelization (e.g. using \
        KMeans) will be dervied using significantly fewer sub_grid, offering run-time speedup.

        In the routine below, some variables and function names refer to a 'sparse_source_'. This term describes a \
        source-pixel that we have paired to a sub_coordinate using the sparse_grid grid_coords of image_grid data_to_pixels. Thus, it may not \
        actually be that sub_coordinate's closest source-pixel (the routine will eventually determine this).

        Parameters
        ----------
        sub_coordinates : [(float, float)]
            The x and y sub_grid to be matched to the source_pixel centers.
        image_total : int
            The total number of image_grid data_to_pixels in the image_grid.
        sub_total : int
            The total number of sub_grid data_to_pixels in the image_grid sub_grid-grid_coords.
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
        image_sub_to_source : [int, int]
            The index in source_pixel_centers each image_grid and sub_grid-image_coordinate is matched with. (e.g. if the fifth
            sub_coordinate of the third image_grid pixel is closest to the 3rd source-pixel in source_pixel_centers,
            image_sub_to_source[2,4] = 2).

     """

    image_sub_to_source = np.zeros((image_total, sub_total))

    for image_index in range(len(sub_coordinates)):
        sub_index = 0
        for sub_coordinate in sub_coordinates[image_index]:

            nearest_sparse = find_nearest_sparse(image_index, image_to_sparse)

            nearest_sparse_source = find_nearest_sparse_source(nearest_sparse, sparse_to_source)

            while True:

                separation_sub_coordinate_and_sparse_source = \
                    find_separation_sub_coordinate_and_nearest_sparse_source(source_centers, sub_coordinate,
                                                                             nearest_sparse_source)

                neighboring_source_index, separation_sub_coordinate_and_neighboring_source = \
                    find_separation_and_nearest_neighboring_source(sub_coordinate, source_centers,
                                                                   source_neighbors[nearest_sparse_source])

                if separation_sub_coordinate_and_sparse_source < separation_sub_coordinate_and_neighboring_source:
                    break
                else:
                    nearest_sparse_source = neighboring_source_index

            # If this pixel is closest to the original pixel, it has been paired successfully with its nearest neighbor.
            image_sub_to_source[image_index, sub_index] = nearest_sparse_source
            sub_index += 1

    return image_sub_to_source

def find_nearest_sparse(image_index, image_to_sparse):
    return image_to_sparse[image_index]

def find_nearest_sparse_source(nearest_sparse, sparse_to_source):
    return sparse_to_source[nearest_sparse]

def find_separation_sub_coordinate_and_nearest_sparse_source(source_centers, sub_coordinate, source_pixel):
    nearest_sparse_source_pixel_center = source_centers[source_pixel]
    return compute_squared_separation(sub_coordinate, nearest_sparse_source_pixel_center)

def find_separation_and_nearest_neighboring_source(sub_coordinate, source_centers, source_neighbors):
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

