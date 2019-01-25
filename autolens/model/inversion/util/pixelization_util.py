import numpy as np
from autolens import decorator_util

@decorator_util.jit()
def rectangular_neighbors_from_shape(shape):
    """Compute the neighbors of every pixel as a list of the pixel index's each pixel shares a vertex with.

    The uniformity of the rectangular grid's geometry is used to compute this.
    """


    pixels = shape[0]*shape[1]

    pixel_neighbors = -1 * np.ones(shape=(pixels, 4))
    pixel_neighbors_size = np.zeros(pixels)

    pixel_neighbors, pixel_neighbors_size = compute_corner_neighbors(pixel_neighbors, pixel_neighbors_size,
                                                                     shape, pixels)
    pixel_neighbors, pixel_neighbors_size = compute_top_edge_neighbors(pixel_neighbors, pixel_neighbors_size,
                                                                       shape, pixels)
    pixel_neighbors, pixel_neighbors_size = compute_left_edge_neighbors(pixel_neighbors, pixel_neighbors_size,
                                                                        shape, pixels)
    pixel_neighbors, pixel_neighbors_size = compute_right_edge_neighbors(pixel_neighbors, pixel_neighbors_size,
                                                                         shape, pixels)
    pixel_neighbors, pixel_neighbors_size = compute_bottom_edge_neighbors(pixel_neighbors, pixel_neighbors_size,
                                                                          shape, pixels)
    pixel_neighbors, pixel_neighbors_size = compute_central_neighbors(pixel_neighbors, pixel_neighbors_size,
                                                                      shape, pixels)

    return pixel_neighbors, pixel_neighbors_size

@decorator_util.jit()
def compute_corner_neighbors(pixel_neighbors, pixel_neighbors_size, shape, pixels):

    pixel_neighbors[0, 0:2] = np.array([1, shape[1]])
    pixel_neighbors_size[0] = 2

    pixel_neighbors[shape[1] - 1, 0:2] = np.array([shape[1] - 2, shape[1] + shape[1] - 1])
    pixel_neighbors_size[shape[1] - 1] = 2

    pixel_neighbors[pixels - shape[1], 0:2] = np.array([pixels - shape[1] * 2,
                                                        pixels - shape[1] + 1])
    pixel_neighbors_size[pixels - shape[1]] = 2

    pixel_neighbors[pixels - 1, 0:2] = np.array([pixels - shape[1] - 1, pixels - 2])
    pixel_neighbors_size[pixels - 1] = 2

    return pixel_neighbors, pixel_neighbors_size

@decorator_util.jit()
def compute_top_edge_neighbors(pixel_neighbors, pixel_neighbors_size, shape, pixels):

    for pix in range(1, shape[1] - 1):
        pixel_index = pix
        pixel_neighbors[pixel_index, 0:3] = np.array([pixel_index - 1, pixel_index + 1,
                                                      pixel_index + shape[1]])
        pixel_neighbors_size[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_size

@decorator_util.jit()
def compute_left_edge_neighbors(pixel_neighbors, pixel_neighbors_size, shape, pixels):

    for pix in range(1, shape[0] - 1):
        pixel_index = pix * shape[1]
        pixel_neighbors[pixel_index, 0:3] = np.array([pixel_index - shape[1], pixel_index + 1,
                                                      pixel_index + shape[1]])
        pixel_neighbors_size[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_size

@decorator_util.jit()
def compute_right_edge_neighbors(pixel_neighbors, pixel_neighbors_size, shape, pixels):

    for pix in range(1, shape[0] - 1):
        pixel_index = pix * shape[1] + shape[1] - 1
        pixel_neighbors[pixel_index, 0:3] = np.array([pixel_index - shape[1], pixel_index - 1,
                                                      pixel_index + shape[1]])
        pixel_neighbors_size[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_size

@decorator_util.jit()
def compute_bottom_edge_neighbors(pixel_neighbors, pixel_neighbors_size, shape, pixels):

    for pix in range(1, shape[1] - 1):
        pixel_index = pixels - pix - 1
        pixel_neighbors[pixel_index, 0:3] = np.array([pixel_index - shape[1], pixel_index - 1,
                                                      pixel_index + 1])
        pixel_neighbors_size[pixel_index] = 3

    return pixel_neighbors, pixel_neighbors_size

@decorator_util.jit()
def compute_central_neighbors(pixel_neighbors, pixel_neighbors_size, shape, pixels):

    for x in range(1, shape[0] - 1):
        for y in range(1, shape[1] - 1):
            pixel_index = x * shape[1] + y
            pixel_neighbors[pixel_index, 0:4] = np.array([pixel_index - shape[1], pixel_index - 1,
                                                          pixel_index + 1, pixel_index + shape[1]])
            pixel_neighbors_size[pixel_index] = 4

    return pixel_neighbors, pixel_neighbors_size

@decorator_util.jit()
def voronoi_neighbors_from_pixels_and_ridge_points(pixels, ridge_points):
    """Compute the neighbors of every pixel as a list of the pixel index's each pixel shares a vertex with.

    The ridge points of the Voronoi grid are used to derive this.

    Parameters
    ----------
    ridge_points : scipy.spatial.Voronoi.ridge_points
        Each Voronoi-ridge (two indexes representing a pixel mapping_matrix).
    """

    pixel_neighbors_size = np.zeros(shape=(pixels))

    for ridge_index in range(ridge_points.shape[0]):
        pair0 = ridge_points[ridge_index, 0]
        pair1 = ridge_points[ridge_index, 1]
        pixel_neighbors_size[pair0] += 1
        pixel_neighbors_size[pair1] += 1

    pixel_neighbors_index = np.zeros(shape=(pixels))
    pixel_neighbors = -1 * np.ones(shape=(pixels, int(np.max(pixel_neighbors_size))))

    for ridge_index in range(ridge_points.shape[0]):
        pair0 = ridge_points[ridge_index, 0]
        pair1 = ridge_points[ridge_index, 1]
        pixel_neighbors[pair0, int(pixel_neighbors_index[pair0])] = pair1
        pixel_neighbors[pair1, int(pixel_neighbors_index[pair1])] = pair0
        pixel_neighbors_index[pair0] += 1
        pixel_neighbors_index[pair1] += 1

    return pixel_neighbors, pixel_neighbors_size