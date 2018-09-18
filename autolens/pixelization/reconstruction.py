from autolens import exc
import numpy as np
import numba


class Reconstructor(object):

    def __init__(self, mapping_matrix, regularization_matrix):
        """The matrices and mappings used to linearly invert and fit a data-set.

        Parameters
        -----------
        mapping_matrix : ndarray
            The matrix representing the mapping_matrix between reconstructed_image-pixels and data_vector-pixels.
        regularization_matrix : ndarray
            The matrix defining how the reconstructed_image's pixels are regularized with one another when fitting the
            data_vector.
        """
        self.mapping_matrix = mapping_matrix
        self.regularization_matrix = regularization_matrix

    def reconstruction_from_reconstructor_and_data(self, image, noise, convolver):
        """Fit the masked_image data using the inversion."""

        blurred_mapping_matrix = convolver.convolve_mapping_matrix(self.mapping_matrix)
        curvature_matrix = self.curvature_matrix_from_blurred_mapping_matrix_jit(blurred_mapping_matrix, noise)
        data_vector = self.data_vector_from_blurred_mapping_matrix_and_data_jit(blurred_mapping_matrix, image, noise)
        curvature_reg_matrix = np.add(curvature_matrix, self.regularization_matrix)
        solution_vector = np.linalg.solve(curvature_reg_matrix, data_vector)

        return Reconstruction(blurred_mapping_matrix, self.regularization_matrix, curvature_matrix,
                              curvature_reg_matrix, solution_vector)

    def curvature_matrix_from_blurred_mapping_matrix_jit(self, blurred_mapping, noise_vector):
        flist = np.zeros(blurred_mapping.shape[0])
        iflist = np.zeros(blurred_mapping.shape[0], dtype='int')
        return self.curvature_matrix_from_blurred_mapping_jitted(blurred_mapping, noise_vector, flist, iflist)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def curvature_matrix_from_blurred_mapping_jitted(blurred_mapping, noise_vector, flist, iflist):

        mapping_shape = blurred_mapping.shape

        covariance_matrix = np.zeros((mapping_shape[1], mapping_shape[1]))

        for image_index in range(mapping_shape[0]):
            index=0
            for pix_index in range(mapping_shape[1]):
                if blurred_mapping[image_index, pix_index] > 0.0:
                    index += 1
                    flist[index] = blurred_mapping[image_index, pix_index] / noise_vector[image_index]
                    iflist[index] = pix_index

            if index > 0:
                for i1 in range(index+1):
                    for j1 in range(index+1):
                        ix = iflist[i1]
                        iy = iflist[j1]
                        covariance_matrix[ix, iy] += flist[i1]*flist[j1]

        for i in range(mapping_shape[1]):
            for j in range(mapping_shape[1]):
                covariance_matrix[i, j] = covariance_matrix[j, i]

        return covariance_matrix

    def data_vector_from_blurred_mapping_matrix_and_data_jit(self, blurred_mapping_matrix, image_vector, noise_vector):
        """ Compute the curvature_matrix matrix directly - used to integration_old test that our curvature_matrix matrix generator approach
        truly works."""
        return self.data_vector_from_blurred_mapping_matrix_and_data_jitted(blurred_mapping_matrix, image_vector, noise_vector)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def data_vector_from_blurred_mapping_matrix_and_data_jitted(blurred_mapping_matrix, image_vector, noise_vector):
        """ Compute the curvature_matrix matrix directly - used to integration_old test that our curvature_matrix matrix generator approach
        truly works."""

        mapping_shape = blurred_mapping_matrix.shape

        data_vector = np.zeros(mapping_shape[1])

        for image_index in range(mapping_shape[0]):
            for pix_index in range(mapping_shape[1]):
                data_vector[pix_index] += image_vector[image_index] * \
                                          blurred_mapping_matrix[image_index, pix_index] / (noise_vector[image_index] ** 2.0)

        return data_vector


class Reconstruction(object):

    def __init__(self, blurred_mapping_matrix, regularization_matrix, curvature_matrix,
                 curvature_reg_matrix, solution_vector):
        """The matrices, mappings which have been used to linearly invert and fit a data-set.

        Parameters
        -----------
        blurred_mapping_matrix : ndarray | None
            The matrix representing the mapping_matrix between reconstructed_image-pixels and data-pixels, including a \
            blurring operation (f).
        regularization_matrix : ndarray | None
            The matrix defining how the reconstructed_image's pixels are regularized with one another (H).
        curvature_matrix : ndarray | None
            The curvature_matrix between each reconstructed_image pixel and all other reconstructed_image pixels (F).
        curvature_reg_matrix : ndarray | None
            The curvature_matrix + regularizationo matrix.
        reconstructed_image : ndarray | None
            The vector containing the reconstructed fit of the data.
        """
        self.blurred_mapping_matrix = blurred_mapping_matrix
        self.regularization_matrix = regularization_matrix
        self.curvature_matrix = curvature_matrix
        self.curvature_reg_matrix = curvature_reg_matrix
        self.solution_vector = solution_vector
        self.reconstructed_image = self.reconstructed_image_from_solution_vector_and_blurred_mapping_matrix_jit(
            solution_vector, blurred_mapping_matrix)
    
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def reconstructed_image_from_solution_vector_and_blurred_mapping_matrix_jit(solution_vector, blurred_mapping_matrix):
        """ Map the reconstructed_image pix s_vector back to the masked_image-plane to compute the pixelization's model-masked_image.
        """
        reconstructed_image = np.zeros(blurred_mapping_matrix.shape[0])
        for i in range(blurred_mapping_matrix.shape[0]):
            for j in range(solution_vector.shape[0]):
                reconstructed_image[i] += solution_vector[j] * blurred_mapping_matrix[i, j]

        return reconstructed_image

    @property
    def regularization_term(self):
        """ Compute the regularization_matrix term of a pixelization's Bayesian likelihood function. This represents the sum \
         of the difference in fluxes between every pair of neighboring pixels. This is computed as:

         s_T * H * s = s_vector.T * regularization_matrix_const * s_vector

         The term is referred to as 'G_l' in Warren & Dye 2003, Nightingale & Dye 2015.

         The above works include the regularization_matrix coefficient (lambda) in this calculation. In PyAutoLens, this is  \
         already in the regularization_matrix matrix and thus included in the matrix multiplication.
         """
        return np.matmul(self.solution_vector.T, np.matmul(self.regularization_matrix, self.solution_vector))

    @property
    def log_det_curvature_reg_matrix_term(self):
        return self.log_determinant_of_matrix_cholesky(self.curvature_reg_matrix)

    @property
    def log_det_regularization_matrix_term(self):
        return self.log_determinant_of_matrix_cholesky(self.regularization_matrix)

    @staticmethod
    def log_determinant_of_matrix_cholesky(matrix):
        """There are two terms in the pixelization's Bayesian likelihood function which require the log determinant of \
        a matrix. These are (Nightingale & Dye 2015, Nightingale, Dye and Massey 2018):

        ln[det(F + H)] = ln[det(cov_reg_matrix)]
        ln[det(H)]     = ln[det(regularization_matrix_const)]

        The cov_reg_matrix is positive-definite, which means its log_determinant can be computed efficiently \
        (compared to using np.det) by using a Cholesky decomposition first and summing the log of each diagonal term.

        Parameters
        -----------
        matrix : ndarray
            The positive-definite matrix the log determinant is computed for.
        """
        try:
            return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(matrix))))
        except np.linalg.LinAlgError:
            raise exc.ReconstructionException()