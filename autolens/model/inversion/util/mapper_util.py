import numpy as np
import numba

@numba.jit(nopython=True, parallel=True)
def mapping_matrix_from_sub_to_pix(sub_to_pix, pixels, regular_pixels, sub_to_regular, sub_grid_fraction):
    """Computes the mapping matrix, by iterating over the known mappings between the sub-grid and pixelization.

    Parameters
    -----------
    sub_to_pix : ndarray
        The mappings between the observed regular's sub-pixels and pixelization's pixels.
    pixels : int
        The number of pixels in the pixelization.
    regular_pixels : int
        The number of image pixels in the observed image and thus on the regular grid.
    sub_to_regular : ndarray
        The mappings between the observed regular's sub-pixels and observed regular's pixels.
    sub_grid_fraction : float
        The fractional area each sub-pixel takes up in an regular-pixel.
    """

    mapping_matrix = np.zeros((regular_pixels, pixels))

    for sub_index in range(sub_to_regular.shape[0]):
        mapping_matrix[sub_to_regular[sub_index], sub_to_pix[sub_index]] += sub_grid_fraction

    return mapping_matrix

@numba.jit(nopython=True, parallel=True)
def voronoi_regular_to_pix_from_grids_and_geometry(regular_grid, regular_to_nearest_regular_pix, pixel_centres,
                                                   pixel_neighbors, pixel_neighbors_size):
    """ Compute the mappings between a set of sub-maskedimage pixels and pixels, using the maskedimage's traced \
    pix-plane sub-grid and the pixel centers. This uses the pix-neighbors to perform a graph \
    search when pairing pixels, for efficiency.

    For the Voronoi pixelizations, a cluster set of 'cluster-pixels' are used to determine the pixelization. \
    These provide the mappings between only a sub-set of sub-pixels / maskedimage-pixels and pixels.

    To determine the complete set of sub-pixel to pixel mappings, we must therefore pair every sub-pixel to \
    its nearest pixel (using the sub-pixel's pix-plane coordinate and pixel center). Using a full \
    nearest neighbor search to do this is slow, thus the pixel neighbors (derived via the Voronoi grid) \
    is used to localize each nearest neighbor search.

    In this routine, some variables and function names refer to a 'cluster_pix_'. This term describes a \
    pixel that we have paired to a sub_coordinate using the cluster_coordinate of an maskedimage coordinate. \
    Thus, it may not actually be that sub_coordinate's closest pixel (the routine will eventually
    determine this).

    Parameters
    ----------
    coordinate : [float, float]
        The x and y pix sub-grid grid which are to be matched with their closest pixels.
    nearest_pixel : int
        The nearest pixel defined on the cluster-pixel grid.
    cluster_to_pixelization : [int]
        The mapping_matrix between every cluster-pixel and pixel (e.g. if the fifth pixel maps to \
        the 3rd cluster_pixel, cluster_to_pix[4] = 2).
     """

    regular_to_pix = np.zeros((regular_grid.shape[0]))

    for regular_index in range(regular_grid.shape[0]):

        nearest_pix_pixel_index = regular_to_nearest_regular_pix[regular_index]

        while True:

            nearest_pix_pixel_center = pixel_centres[nearest_pix_pixel_index]

            sub_to_nearest_pix_distance = (regular_grid[regular_index, 0] - nearest_pix_pixel_center[0]) ** 2 + \
                                          (regular_grid[regular_index, 1] - nearest_pix_pixel_center[1]) ** 2

            closest_separation_from_pix_neighbor = 1.0e8

            for neighbor_index in range(pixel_neighbors_size[nearest_pix_pixel_index]):

                neighbor = pixel_neighbors[nearest_pix_pixel_index, neighbor_index]

                separation_from_neighbor = (regular_grid[regular_index, 0] - pixel_centres[neighbor, 0]) ** 2 + \
                                           (regular_grid[regular_index, 1] - pixel_centres[neighbor, 1]) ** 2

                if separation_from_neighbor < closest_separation_from_pix_neighbor:
                    closest_separation_from_pix_neighbor = separation_from_neighbor
                    closest_neighbor_index = neighbor_index

            neighboring_pix_pixel_index = pixel_neighbors[nearest_pix_pixel_index, closest_neighbor_index]
            sub_to_neighboring_pix_distance = closest_separation_from_pix_neighbor

            if sub_to_nearest_pix_distance <= sub_to_neighboring_pix_distance:
                regular_to_pix[regular_index] = nearest_pix_pixel_index
                break
            else:
                nearest_pix_pixel_index = neighboring_pix_pixel_index

    return regular_to_pix

@numba.jit(nopython=True, parallel=True)
def voronoi_sub_to_pix_from_grids_and_geometry(sub_grid, regular_to_nearest_regular_pix, sub_to_regular, pixel_centres,
                                               pixel_neighbors, pixel_neighbors_size):
    """ Compute the mappings between a set of sub-maskedimage pixels and pixels, using the maskedimage's traced \
    pix-plane sub-grid and the pixel centers. This uses the pix-neighbors to perform a graph \
    search when pairing pixels, for efficiency.

    For the Voronoi pixelizations, a cluster set of 'cluster-pixels' are used to determine the pixelization. \
    These provide the mappings between only a sub-set of sub-pixels / maskedimage-pixels and pixels.

    To determine the complete set of sub-pixel to pixel mappings, we must therefore pair every sub-pixel to \
    its nearest pixel (using the sub-pixel's pix-plane coordinate and pixel center). Using a full \
    nearest neighbor search to do this is slow, thus the pixel neighbors (derived via the Voronoi grid) \
    is used to localize each nearest neighbor search.

    In this routine, some variables and function names refer to a 'cluster_pix_'. This term describes a \
    pixel that we have paired to a sub_coordinate using the cluster_coordinate of an maskedimage coordinate. \
    Thus, it may not actually be that sub_coordinate's closest pixel (the routine will eventually
    determine this).

    Parameters
    ----------
    coordinate : [float, float]
        The x and y pix sub-grid grid which are to be matched with their closest pixels.
    nearest_pixel : int
        The nearest pixel defined on the cluster-pixel grid.
    cluster_to_pixelization : [int]
        The mapping_matrix between every cluster-pixel and pixel (e.g. if the fifth pixel maps to \
        the 3rd cluster_pixel, cluster_to_pix[4] = 2).
     """

    sub_to_pix = np.zeros((sub_grid.shape[0]))

    for sub_index in range(sub_grid.shape[0]):

        nearest_pix_pixel_index = regular_to_nearest_regular_pix[sub_to_regular[sub_index]]

        while True:

            nearest_pix_pixel_center = pixel_centres[nearest_pix_pixel_index]

            sub_to_nearest_pix_distance = (sub_grid[sub_index, 0] - nearest_pix_pixel_center[0]) ** 2 + \
                                          (sub_grid[sub_index, 1] - nearest_pix_pixel_center[1]) ** 2

            closest_separation_from_pix_to_neighbor = 1.0e8

            for neighbor_index in range(pixel_neighbors_size[nearest_pix_pixel_index]):

                neighbor = pixel_neighbors[nearest_pix_pixel_index, neighbor_index]

                separation_from_neighbor = (sub_grid[sub_index, 0] - pixel_centres[neighbor, 0]) ** 2 + \
                                           (sub_grid[sub_index, 1] - pixel_centres[neighbor, 1]) ** 2

                if separation_from_neighbor < closest_separation_from_pix_to_neighbor:
                    closest_separation_from_pix_to_neighbor = separation_from_neighbor
                    closest_neighbor_index = neighbor_index

            neighboring_pix_pixel_index = pixel_neighbors[nearest_pix_pixel_index, closest_neighbor_index]
            sub_to_neighboring_pix_distance = closest_separation_from_pix_to_neighbor

            if sub_to_nearest_pix_distance <= sub_to_neighboring_pix_distance:
                sub_to_pix[sub_index] = nearest_pix_pixel_index
                break
            else:
                nearest_pix_pixel_index = neighboring_pix_pixel_index

    return sub_to_pix