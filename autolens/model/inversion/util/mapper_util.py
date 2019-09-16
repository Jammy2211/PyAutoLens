import numpy as np
from autolens import decorator_util


@decorator_util.jit()
def mapping_matrix_from_sub_mask_1d_index_to_pixelization_1d_index(
    sub_mask_1d_index_to_pixelization_1d_index,
    pixels,
    total_mask_pixels,
    sub_mask_1d_index_to_mask_1d_index,
    sub_fraction,
):
    """Computes the mapping_util matrix, by iterating over the known mappings between the sub-grid and pixelization.

    Parameters
    -----------
    sub_mask_1d_index_to_pixelization_1d_index : ndarray
        The mappings between the observed grid's sub-pixels and pixelization's pixels.
    pixels : int
        The number of pixels in the pixelization.
    total_mask_pixels : int
        The number of datas pixels in the observed datas and thus on the grid.
    sub_mask_1d_index_to_mask_1d_index : ndarray
        The mappings between the observed grid's sub-pixels and observed grid's pixels.
    sub_fraction : float
        The fractional area each sub-pixel takes up in an pixel.
    """

    mapping_matrix = np.zeros((total_mask_pixels, pixels))

    for sub_mask_1d_index in range(sub_mask_1d_index_to_mask_1d_index.shape[0]):
        mapping_matrix[
            sub_mask_1d_index_to_mask_1d_index[sub_mask_1d_index],
            sub_mask_1d_index_to_pixelization_1d_index[sub_mask_1d_index],
        ] += sub_fraction

    return mapping_matrix


@decorator_util.jit()
def voronoi_sub_mask_1d_index_to_pixeliztion_1d_index_from_grids_and_geometry(
    grid,
    mask_1d_index_to_nearest_pixelization_1d_index,
    sub_mask_1d_index_to_mask_1d_index,
    pixel_centres,
    pixel_neighbors,
    pixel_neighbors_size,
):
    """ Compute the mappings between a set of sub-grid pixels and pixelization pixels, using information on \
    how the pixels hosting each sub-pixel map to their closest pixelization pixel on the image-plane pix-grid \
    and the pixelization's pixel centres.

    To determine the complete set of sub-pixel to pixelization pixel mappings, we must pair every sub-pixel to \
    its nearest pixel. Using a full nearest neighbor search to do this is slow, thus the pixel neighbors (derived via \
    the Voronoi grid) are used to localize each nearest neighbor search by using a graph search.

    Parameters
    ----------
    grid : Grid
        The grid of (y,x) arc-second coordinates at the centre of every unmasked pixel, which has been traced to \
        to an irgrid via lens.
    mask_1d_index_to_nearest_pixelization_1d_index : ndarray
        A 1D array that maps every grid pixel to its nearest pix-grid pixel (as determined on the unlensed \
        2D array).
    pixel_centres : (float, float)
        The (y,x) centre of every Voronoi pixel in arc-seconds.
    pixel_neighbors : ndarray
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in \
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : ndarray
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the \
        Voronoi grid.
     """

    sub_mask_1d_index_to_pixeliztion_1d_index = np.zeros((grid.shape[0]))

    for sub_mask_1d_index in range(grid.shape[0]):

        nearest_pixelization_1d_index = mask_1d_index_to_nearest_pixelization_1d_index[
            sub_mask_1d_index_to_mask_1d_index[sub_mask_1d_index]
        ]

        while True:

            nearest_pixelization_pixel_center = pixel_centres[
                nearest_pixelization_1d_index
            ]

            sub_pixel_to_nearest_pixelization_distance = (
                (grid[sub_mask_1d_index, 0] - nearest_pixelization_pixel_center[0]) ** 2
                + (grid[sub_mask_1d_index, 1] - nearest_pixelization_pixel_center[1])
                ** 2
            )

            closest_separation_from_pixelization_to_neighbor = 1.0e8

            for neighbor_pixelization_1d_index in range(
                pixel_neighbors_size[nearest_pixelization_1d_index]
            ):

                neighbor = pixel_neighbors[
                    nearest_pixelization_1d_index, neighbor_pixelization_1d_index
                ]

                separation_from_neighbor = (
                    grid[sub_mask_1d_index, 0] - pixel_centres[neighbor, 0]
                ) ** 2 + (grid[sub_mask_1d_index, 1] - pixel_centres[neighbor, 1]) ** 2

                if (
                    separation_from_neighbor
                    < closest_separation_from_pixelization_to_neighbor
                ):
                    closest_separation_from_pixelization_to_neighbor = (
                        separation_from_neighbor
                    )
                    closest_neighbor_pixelization_1d_index = (
                        neighbor_pixelization_1d_index
                    )

            neighboring_pixelization_1d_index = pixel_neighbors[
                nearest_pixelization_1d_index, closest_neighbor_pixelization_1d_index
            ]
            sub_pixel_to_neighboring_pixelization_distance = (
                closest_separation_from_pixelization_to_neighbor
            )

            if (
                sub_pixel_to_nearest_pixelization_distance
                <= sub_pixel_to_neighboring_pixelization_distance
            ):
                sub_mask_1d_index_to_pixeliztion_1d_index[
                    sub_mask_1d_index
                ] = nearest_pixelization_1d_index
                break
            else:
                nearest_pixelization_1d_index = neighboring_pixelization_1d_index

    return sub_mask_1d_index_to_pixeliztion_1d_index
