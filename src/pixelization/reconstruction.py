import numpy as np
import numba


class Reconstructor(object):

    def __init__(self, mapping, regularization, image_to_pix, sub_to_pix):
        """The matrices and mappings used to linearly invert and fit a data-set.

        Parameters
        -----------
        mapping : ndarray
            The matrix representing the mapping between reconstruction-pixels and weighted_data-pixels.
        regularization : ndarray
            The matrix defining how the reconstruction's pixels are regularized with one another when fitting the
            weighted_data.
        image_to_pix : ndarray
            The mapping between each image-grid pixel and pixelization-grid pixel.
        sub_to_pix : ndarray
            The mapping between each sub-grid pixel and pixelization-grid sub-pixel.
        """
        self.mapping = mapping
        self.mapping_shape = mapping.shape
        self.regularization = regularization
        self.image_to_pix = image_to_pix
        self.sub_to_pix = sub_to_pix

    def reconstruct_image(self, image, noise, convolver):
        """Fit the image data using the inversion."""

        blurred_mapping = convolver.convolve_mapping_matrix(self.mapping)
        # TODO : Use fast routines once ready.
        covariance = self.covariance_matrix_from_blurred_mapping(blurred_mapping, noise)
        weighted_data = self.data_vector_from_blurred_mapping_and_data(blurred_mapping, image, noise)
        cov_reg = covariance + self.regularization
        reconstruction = np.linalg.solve(cov_reg, weighted_data)

        return Reconstruction(weighted_data, blurred_mapping, self.regularization, covariance, cov_reg, reconstruction)

    def covariance_matrix_from_blurred_mapping(self, blurred_mapping, noise_vector):
        """ Compute the covariance matrix directly - used to integration test that our covariance matrix generator approach
        truly works."""

        covariance_matrix = np.zeros((self.mapping_shape[1], self.mapping_shape[1]))

        for i in range(self.mapping_shape[0]):
            for jx in range(self.mapping_shape[1]):
                for jy in range(self.mapping_shape[1]):
                    covariance_matrix[jx, jy] += blurred_mapping[i, jx] * blurred_mapping[i, jy] \
                                                 / (noise_vector[i] ** 2.0)

        return covariance_matrix

    def data_vector_from_blurred_mapping_and_data(self, blurred_mapping, image_vector, noise_vector):
        """ Compute the covariance matrix directly - used to integration test that our covariance matrix generator approach
        truly works."""
        data_vector = np.zeros((self.mapping_shape[1],))

        for i in range(self.mapping_shape[0]):
            for j in range(self.mapping_shape[1]):
                data_vector[j] += image_vector[i] * blurred_mapping[i, j] / (noise_vector[i] ** 2.0)

        return data_vector


class Reconstruction(object):

    def __init__(self, weighted_data, blurred_mapping, regularization, covariance, covariance_regularization,
                 reconstruction):
        """The matrices, mappings which have been used to linearly invert and fit a data-set.

        Parameters
        -----------
        weighted_data : ndarray | None
            The 1D vector representing the data, weighted by its noise in a chi squared sense, which is fitted by the \
            inversion (D).
        blurred_mapping : ndarray | None
            The matrix representing the mapping between reconstruction-pixels and data-pixels, including a \
            blurring operation (f).
        regularization : ndarray | None
            The matrix defining how the reconstruction's pixels are regularized with one another (H).
        covariance : ndarray | None
            The covariance between each reconstruction pixel and all other reconstruction pixels (F).
        covariance_regularization : ndarray | None
            The covariance + regularizationo matrix.
        reconstruction : ndarray | None
            The vector containing the reconstructed fit of the data.
        """
        self.weighted_data = weighted_data
        self.blurred_mapping = blurred_mapping
        self.regularization = regularization
        self.covariance = covariance
        self.covariance_regularization = covariance_regularization
        self.reconstruction = reconstruction

    def model_image_from_reconstruction(self):
        """ Map the reconstruction pix s_vector back to the image-plane to compute the pixelization's model-image.
        """
        model_image = np.zeros(self.blurred_mapping.shape[0])
        for i in range(self.blurred_mapping.shape[0]):
            for j in range(len(self.reconstruction)):
                model_image[i] += self.reconstruction[j] * self.blurred_mapping[i, j]

        return model_image

    # TODO : Speed this up using pix_pixel neighbors list to skip sparsity (see regularization matrix calculation)
    def regularization_term_from_reconstruction(self):
        """ Compute the regularization term of a pixelization's Bayesian likelihood function. This represents the sum \
         of the difference in fluxes between every pair of neighboring pixels. This is computed as:

         s_T * H * s = s_vector.T * regularization_matrix_const * s_vector

         The term is referred to as 'G_l' in Warren & Dye 2003, Nightingale & Dye 2015.

         The above works include the regularization coefficient (lambda) in this calculation. In PyAutoLens, this is  \
         already in the regularization matrix and thus included in the matrix multiplication.
         """
        return np.matmul(self.reconstruction.T, np.matmul(self.regularization, self.reconstruction))

    # TODO : Cholesky decomposition can also use pixel neighbors list to skip sparsity.
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
        return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(matrix))))