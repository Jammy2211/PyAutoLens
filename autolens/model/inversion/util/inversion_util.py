import numba
import numpy as np

@numba.jit(nopython=True, parallel=True)
def data_vector_from_blurred_mapping_matrix_and_data(blurred_mapping_matrix, image, noise_map):
    """ Compute the curvature_matrix matrix directly - used to integration_old test that our curvature_matrix matrix generator approach
    truly works."""

    mapping_shape = blurred_mapping_matrix.shape

    data_vector = np.zeros(mapping_shape[1])

    for image_index in range(mapping_shape[0]):
        for pix_index in range(mapping_shape[1]):
            data_vector[pix_index] += image[image_index] * \
                                      blurred_mapping_matrix[image_index, pix_index] / (noise_map[image_index] ** 2.0)

    return data_vector

def curvature_matrix_from_blurred_mapping_matrix(blurred_mapping_matrix, noise_map):
    flist = np.zeros(blurred_mapping_matrix.shape[0])
    iflist = np.zeros(blurred_mapping_matrix.shape[0], dtype='int')
    return curvature_matrix_from_blurred_mapping_matrix_jit(blurred_mapping_matrix, noise_map, flist, iflist)

@numba.jit(nopython=True, parallel=True)
def curvature_matrix_from_blurred_mapping_matrix_jit(blurred_mapping_matrix, noise_map, flist, iflist):

    curvature_matrix = np.zeros((blurred_mapping_matrix.shape[1], blurred_mapping_matrix.shape[1]))

    for image_index in range(blurred_mapping_matrix.shape[0]):
        index = 0
        for pixel_index in range(blurred_mapping_matrix.shape[1]):
            if blurred_mapping_matrix[image_index, pixel_index] > 0.0:
                flist[index] = blurred_mapping_matrix[image_index, pixel_index] / noise_map[image_index]
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

@numba.jit(nopython=True, parallel=True)
def reconstructed_data_vector_from_blurred_mapping_matrix_and_solution_vector(blurred_mapping_matrix, solution_vector):
    """ Map the reconstructed_image pix s_vector back to the masked_image-plane to compute the inversion's model-masked_image.
    """
    reconstructed_data_vector = np.zeros(blurred_mapping_matrix.shape[0])
    for i in range(blurred_mapping_matrix.shape[0]):
        for j in range(solution_vector.shape[0]):
            reconstructed_data_vector[i] += solution_vector[j] * blurred_mapping_matrix[i, j]

    return reconstructed_data_vector