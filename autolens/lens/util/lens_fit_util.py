import numpy as np


def likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(chi_squared,
                                                                                                regularization_term,
                                                                                                noise_normalization):
    """Compute the likelihood of an inversion's fit to the datas, including a regularization term which \
    comes from an inversion:

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared : float
        The chi-squared term of the inversion's fit to the observed datas.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    noise_normalization : float
        The normalization noise_map-term for the observed datas's noise-map.
    """
    return -0.5 * (chi_squared + regularization_term + noise_normalization)


def evidence_from_inversion_terms(chi_squared, regularization_term, log_curvature_regularization_term,
                                  log_regularization_term, noise_normalization):
    """Compute the evidence of an inversion's fit to the datas, where the evidence includes a number of \
    terms which quantify the complexity of an inversion's reconstruction (see the *inversion* module):

    Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                       Log(Regularization_Matrix_Term) + Noise_Term]

    Parameters
    ----------
    chi_squared : float
        The chi-squared term of the inversion's fit to the observed datas.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    log_curvature_regularization_term : float
        The log of the determinant of the sum of the curvature and regularization matrices.
    log_regularization_term : float
        The log of the determinant o the regularization matrix.
    noise_normalization : float
        The normalization noise_map-term for the observed datas's noise-map.
    """
    return -0.5 * (chi_squared + regularization_term + log_curvature_regularization_term - log_regularization_term
                   + noise_normalization)
