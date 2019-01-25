from autolens import decorator_util
import numpy as np

@decorator_util.jit()
def data_vector_from_blurred_mapping_matrix_and_data(blurred_mapping_matrix, image_1d, noise_map_1d):
    """Compute the hyper vector *D* from a blurred mapping matrix *f* and the 1D image *d* and 1D noise-map *\sigma* \
    (see Warren & Dye 2003).
    
    Parameters
    -----------
    blurred_mapping_matrix : ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    image_1d : ndarray
        Flattened 1D array of the observed image the inversion is fitting.
    noise_map_1d : ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """

    mapping_shape = blurred_mapping_matrix.shape

    data_vector = np.zeros(mapping_shape[1])

    for image_index in range(mapping_shape[0]):
        for pix_index in range(mapping_shape[1]):
            data_vector[pix_index] += image_1d[image_index] * \
                                      blurred_mapping_matrix[image_index, pix_index] / (noise_map_1d[image_index] ** 2.0)

    return data_vector

def curvature_matrix_from_blurred_mapping_matrix(blurred_mapping_matrix, noise_map_1d):
    """Compute the curvature matrix *F* from a blurred mapping matrix *f* and the 1D noise-map *\sigma* \
     (see Warren & Dye 2003).

    Parameters
    -----------
    blurred_mapping_matrix : ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    noise_map_1d : ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """

    flist = np.zeros(blurred_mapping_matrix.shape[0])
    iflist = np.zeros(blurred_mapping_matrix.shape[0], dtype='int')
    return curvature_matrix_from_blurred_mapping_matrix_jit(blurred_mapping_matrix, noise_map_1d, flist, iflist)

@decorator_util.jit()
def curvature_matrix_from_blurred_mapping_matrix_jit(blurred_mapping_matrix, noise_map_1d, flist, iflist):
    """Compute the curvature matrix *F* from a blurred mapping matrix *f* and the 1D noise-map *\sigma* \
    (see Warren & Dye 2003).

    Parameters
    -----------
    blurred_mapping_matrix : ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    noise_map_1d : ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    flist : ndarray
        NumPy array of floats used to store mappings for efficienctly calculation.
    iflist : ndarray
        NumPy array of integers used to store mappings for efficienctly calculation.
    """
    curvature_matrix = np.zeros((blurred_mapping_matrix.shape[1], blurred_mapping_matrix.shape[1]))

    for image_index in range(blurred_mapping_matrix.shape[0]):
        index = 0
        for pixel_index in range(blurred_mapping_matrix.shape[1]):
            if blurred_mapping_matrix[image_index, pixel_index] > 0.0:
                flist[index] = blurred_mapping_matrix[image_index, pixel_index] / noise_map_1d[image_index]
                iflist[index] = pixel_index
                index += 1

        if index > 0:
            for i1 in range(index):
                for j1 in range(index):
                    ix = iflist[i1]
                    iy = iflist[j1]
                    curvature_matrix[ix, iy] += flist[i1] * flist[j1]

    for i in range(blurred_mapping_matrix.shape[1]):
        for j in range(blurred_mapping_matrix.shape[1]):
            curvature_matrix[i, j] = curvature_matrix[j, i]

    return curvature_matrix

@decorator_util.jit()
def reconstructed_data_vector_from_blurred_mapping_matrix_and_solution_vector(blurred_mapping_matrix, solution_vector):
    """ Compute the reconstructed hyper vector from the blurrred mapping matrix *f* and solution vector *S*.

    Parameters
    -----------
    blurred_mapping_matrix : ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """
    reconstructed_data_vector = np.zeros(blurred_mapping_matrix.shape[0])
    for i in range(blurred_mapping_matrix.shape[0]):
        for j in range(solution_vector.shape[0]):
            reconstructed_data_vector[i] += solution_vector[j] * blurred_mapping_matrix[i, j]

    return reconstructed_data_vector