import numpy as np

from autolens import exc
from autolens.model.inversion.util import inversion_util

# TODO : Unit test this properly, using a cleverly made mock datas-set

def inversion_from_lensing_image_mapper_and_regularization(image, noise_map, convolver, mapper, regularization):
    return Inversion(image=image, noise_map=noise_map, convolver=convolver, mapper=mapper,
                     regularization=regularization)

class Inversion(object):

    def __init__(self, image, noise_map, convolver, mapper, regularization):
        """The matrices, mappings which have been used to linearly invert and fit_normal a datas-set.

        Parameters
        -----------
        image : ndarray
            Flattened 1D array of the regular the inversion fits.
        noise_map : ndarray
            Flattened 1D array of the noise-map used by the inversion.
        convolver : imaging.convolution.Convolver
            The convolver used to blur the mapping matrix with the PSF.
        mapper : inversion.mappers.Mapper
            The mapping between the regular and pixelization.
        regularization : inversion.regularization.Regularization
            The regularization scheme applied to the pixeliztion for the inversion

        Attributes
        -----------
        blurred_mapping_matrix : ndarray | None
            The matrix representing the mapping_matrix between reconstructed_image-pixels and datas-pixels, including a \
            blurring operation (f).
        regularization_matrix : ndarray | None
            The matrix defining how the reconstructed_image's pixels are regularized with one another (H).
        curvature_matrix : ndarray | None
            The curvature_matrix between each reconstructed_image pixel and all other reconstructed_image pixels (F).
        curvature_reg_matrix : ndarray | None
            The curvature_matrix + regularizationo matrix.
        reconstructed_image : ndarray | None
            The vector containing the reconstructed fit_normal of the datas.
        """

        self.mapper = mapper
        self.regularization = regularization
        self.blurred_mapping_matrix = convolver.convolve_mapping_matrix(mapping_matrix=mapper.mapping_matrix)

        self.data_vector = inversion_util.data_vector_from_blurred_mapping_matrix_and_data(
                blurred_mapping_matrix=self.blurred_mapping_matrix, image=image, noise_map=noise_map)

        self.curvature_matrix = inversion_util.curvature_matrix_from_blurred_mapping_matrix(
                blurred_mapping_matrix=self.blurred_mapping_matrix, noise_map=noise_map)

        self.regularization_matrix = \
            regularization.regularization_matrix_from_pixel_neighbors(pixel_neighbors=mapper.geometry.pixel_neighbors,
                                                            pixel_neighbors_size=mapper.geometry.pixel_neighbors_size)
        self.curvature_reg_matrix = np.add(self.curvature_matrix, self.regularization_matrix)
        self.solution_vector = np.linalg.solve(self.curvature_reg_matrix, self.data_vector)

    @property
    def reconstructed_data(self):
        return self.mapper.grids.regular.scaled_array_from_array_1d(np.asarray(self.reconstructed_data_vector))

    @property
    def reconstructed_data_vector(self):
        return inversion_util.reconstructed_data_vector_from_blurred_mapping_matrix_and_solution_vector(
            self.blurred_mapping_matrix, self.solution_vector)

    @property
    def regularization_term(self):
        """ Compute the regularization_matrix term of a inversion's Bayesian likelihood function. This represents the sum \
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
        """There are two terms in the inversion's Bayesian likelihood function which require the log determinant of \
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
            raise exc.InversionException()