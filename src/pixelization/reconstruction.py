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

        blurred_mapping = convolver.convolve_mapping_matrix_jit(self.mapping)
        covariance = self.covariance_matrix_from_blurred_mapping_jit(blurred_mapping, noise)
        data_vector = self.data_vector_from_blurred_mapping_and_data_jit(blurred_mapping, image, noise)
        cov_reg = np.add(covariance, self.regularization)
        reconstruction = np.linalg.solve(cov_reg, data_vector)

        return Reconstruction(data_vector, blurred_mapping, self.regularization, covariance, cov_reg, reconstruction)

    def covariance_matrix_from_blurred_mapping_jit(self, blurred_mapping, noise_vector):
        flist = np.zeros(blurred_mapping.shape[0])
        iflist = np.zeros(blurred_mapping.shape[0], dtype='int')
        return self.covariance_matrix_from_blurred_mapping_jitted(blurred_mapping, noise_vector, flist, iflist)

    @staticmethod
    @numba.jit(nopython=True)
    def covariance_matrix_from_blurred_mapping_jitted(blurred_mapping, noise_vector, flist, iflist):

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

    def data_vector_from_blurred_mapping_and_data_jit(self, blurred_mapping, image_vector, noise_vector):
        """ Compute the covariance matrix directly - used to integration test that our covariance matrix generator approach
        truly works."""
        return self.data_vector_from_blurred_mapping_and_data_jitted(blurred_mapping, image_vector, noise_vector)

    @staticmethod
    @numba.jit(nopython=True)
    def data_vector_from_blurred_mapping_and_data_jitted(blurred_mapping, image_vector, noise_vector):
        """ Compute the covariance matrix directly - used to integration test that our covariance matrix generator approach
        truly works."""

        mapping_shape = blurred_mapping.shape

        data_vector = np.zeros(mapping_shape[1])

        for image_index in range(mapping_shape[0]):
            for pix_index in range(mapping_shape[1]):
                data_vector[pix_index] += image_vector[image_index] * \
                                          blurred_mapping[image_index, pix_index] / (noise_vector[image_index] ** 2.0)

        return data_vector


class Reconstruction(object):

    def __init__(self, data_vector, blurred_mapping, regularization, covariance, covariance_regularization,
                 reconstruction):
        """The matrices, mappings which have been used to linearly invert and fit a data-set.

        Parameters
        -----------
        data_vector : ndarray | None
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
        self.weighted_data = data_vector
        self.blurred_mapping = blurred_mapping
        self.regularization = regularization
        self.covariance = covariance
        self.covariance_regularization = covariance_regularization
        self.reconstruction = reconstruction

    def model_image_from_reconstruction_jit(self):
        """ Map the reconstruction pix s_vector back to the image-plane to compute the pixelization's model-image.
        """
        return self.model_image_from_reconstruction_jitted(self.reconstruction, self.blurred_mapping)
    
    @staticmethod
    @numba.jit(nopython=True)
    def model_image_from_reconstruction_jitted(reconstruction, blurred_mapping):
        """ Map the reconstruction pix s_vector back to the image-plane to compute the pixelization's model-image.
        """
        model_image = np.zeros(blurred_mapping.shape[0])
        for i in range(blurred_mapping.shape[0]):
            for j in range(reconstruction.shape[0]):
                model_image[i] += reconstruction[j] * blurred_mapping[i, j]

        return model_image

    def regularization_term_from_reconstruction(self):
        """ Compute the regularization term of a pixelization's Bayesian likelihood function. This represents the sum \
         of the difference in fluxes between every pair of neighboring pixels. This is computed as:

         s_T * H * s = s_vector.T * regularization_matrix_const * s_vector

         The term is referred to as 'G_l' in Warren & Dye 2003, Nightingale & Dye 2015.

         The above works include the regularization coefficient (lambda) in this calculation. In PyAutoLens, this is  \
         already in the regularization matrix and thus included in the matrix multiplication.
         """
        return np.matmul(self.reconstruction.T, np.matmul(self.regularization, self.reconstruction))

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