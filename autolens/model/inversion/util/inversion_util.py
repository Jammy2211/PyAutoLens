from autolens import decorator_util
import numpy as np


@decorator_util.jit()
def data_vector_from_blurred_mapping_matrix_and_data(
    blurred_mapping_matrix, image_1d, noise_map_1d
):
    """Compute the hyper_galaxies vector *D* from a blurred mapping_util matrix *f* and the 1D image *d* and 1D noise-map *\sigma* \
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

    for mask_1d_index in range(mapping_shape[0]):
        for pixelization_1d_index in range(mapping_shape[1]):
            data_vector[pixelization_1d_index] += (
                image_1d[mask_1d_index]
                * blurred_mapping_matrix[mask_1d_index, pixelization_1d_index]
                / (noise_map_1d[mask_1d_index] ** 2.0)
            )

    return data_vector


def curvature_matrix_from_blurred_mapping_matrix(blurred_mapping_matrix, noise_map_1d):
    """Compute the curvature matrix *F* from a blurred mapping_util matrix *f* and the 1D noise-map *\sigma* \
     (see Warren & Dye 2003).

    Parameters
    -----------
    blurred_mapping_matrix : ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    noise_map_1d : ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """

    flist = np.zeros(blurred_mapping_matrix.shape[1])
    iflist = np.zeros(blurred_mapping_matrix.shape[1], dtype="int")
    return curvature_matrix_from_blurred_mapping_matrix_jit(
        blurred_mapping_matrix, noise_map_1d, flist, iflist
    )


@decorator_util.jit()
def curvature_matrix_from_blurred_mapping_matrix_jit(
    blurred_mapping_matrix, noise_map_1d, flist, iflist
):
    """Compute the curvature matrix *F* from a blurred mapping_util matrix *f* and the 1D noise-map *\sigma* \
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
    curvature_matrix = np.zeros(
        (blurred_mapping_matrix.shape[1], blurred_mapping_matrix.shape[1])
    )

    for mask_1d_index in range(blurred_mapping_matrix.shape[0]):
        index = 0
        for pixelization_1d_index in range(blurred_mapping_matrix.shape[1]):
            if blurred_mapping_matrix[mask_1d_index, pixelization_1d_index] > 0.0:
                flist[index] = (
                    blurred_mapping_matrix[mask_1d_index, pixelization_1d_index]
                    / noise_map_1d[mask_1d_index]
                )
                iflist[index] = pixelization_1d_index
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
def reconstructed_data_vector_from_blurred_mapping_matrix_and_solution_vector(
    blurred_mapping_matrix, solution_vector
):
    """ Compute the reconstructed hyper_galaxies vector from the blurrred mapping_util matrix *f* and solution vector *S*.

    Parameters
    -----------
    blurred_mapping_matrix : ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """
    reconstructed_data_vector = np.zeros(blurred_mapping_matrix.shape[0])
    for i in range(blurred_mapping_matrix.shape[0]):
        for j in range(solution_vector.shape[0]):
            reconstructed_data_vector[i] += (
                solution_vector[j] * blurred_mapping_matrix[i, j]
            )

    return reconstructed_data_vector


def pixelization_residual_map_from_pixelization_values_and_reconstructed_data_1d(
    pixelization_values,
    reconstructed_data_1d,
    sub_mask_1d_index_to_mask_1d_index,
    pixelization_1d_index_to_all_sub_mask_1d_indexes,
):

    pixelization_residuals = np.zeros(
        shape=len(pixelization_1d_index_to_all_sub_mask_1d_indexes)
    )

    reconstructed_data_1d = reconstructed_data_1d

    for pixelization_1d_index, sub_mask_1d_indexes in enumerate(
        pixelization_1d_index_to_all_sub_mask_1d_indexes
    ):
        for sub_mask_1d_index in sub_mask_1d_indexes:
            mask_1d_index = sub_mask_1d_index_to_mask_1d_index[sub_mask_1d_index]
            residual = (
                reconstructed_data_1d[mask_1d_index]
                - pixelization_values[pixelization_1d_index]
            )
            pixelization_residuals[pixelization_1d_index] += np.abs(residual)

    return pixelization_residuals


def pixelization_normalized_residual_map_from_pixelization_values_and_reconstructed_data_1d(
    pixelization_values,
    reconstructed_data_1d,
    noise_map_1d,
    sub_mask_1d_index_to_mask_1d_index,
    pixelization_1d_index_to_all_sub_mask_1d_indexes,
):

    pixelization_normalized_residuals = np.zeros(
        shape=len(pixelization_1d_index_to_all_sub_mask_1d_indexes)
    )

    reconstructed_data_1d = reconstructed_data_1d

    for pixelization_1d_index, sub_mask_1d_indexes in enumerate(
        pixelization_1d_index_to_all_sub_mask_1d_indexes
    ):
        for sub_mask_1d_index in sub_mask_1d_indexes:
            mask_1d_index = sub_mask_1d_index_to_mask_1d_index[sub_mask_1d_index]
            residual = (
                reconstructed_data_1d[mask_1d_index]
                - pixelization_values[pixelization_1d_index]
            )
            pixelization_normalized_residuals[pixelization_1d_index] += np.abs(
                (residual / noise_map_1d[mask_1d_index])
            )

    return pixelization_normalized_residuals


def pixelization_chi_squared_map_from_pixelization_values_and_reconstructed_data_1d(
    pixelization_values,
    reconstructed_data_1d,
    noise_map_1d,
    sub_mask_1d_index_to_mask_1d_index,
    pixelization_1d_index_to_all_sub_mask_1d_indexes,
):

    pixelization_chi_squareds = np.zeros(
        shape=len(pixelization_1d_index_to_all_sub_mask_1d_indexes)
    )

    reconstructed_data_1d = reconstructed_data_1d

    for pixelization_1d_index, sub_mask_1d_indexes in enumerate(
        pixelization_1d_index_to_all_sub_mask_1d_indexes
    ):
        for sub_mask_1d_index in sub_mask_1d_indexes:
            mask_1d_index = sub_mask_1d_index_to_mask_1d_index[sub_mask_1d_index]
            residual = (
                reconstructed_data_1d[mask_1d_index]
                - pixelization_values[pixelization_1d_index]
            )
            pixelization_chi_squareds[pixelization_1d_index] += (
                residual / noise_map_1d[mask_1d_index]
            ) ** 2.0

    return pixelization_chi_squareds
