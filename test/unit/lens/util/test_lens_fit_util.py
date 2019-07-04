from autolens.lens.util import lens_fit_util as util

class TestInversionEvidence:

    def test__simple_values(self):

        likelihood_with_regularization_terms = \
            util.likelihood_with_regularization_from_chi_squared_regularization_term_and_noise_normalization(
                chi_squared=3.0, regularization_term=6.0, noise_normalization=2.0)

        assert likelihood_with_regularization_terms == -0.5 * (3.0 + 6.0 + 2.0)

        evidences = util.evidence_from_inversion_terms(
            chi_squared=3.0, regularization_term=6.0, log_curvature_regularization_term=9.0,
            log_regularization_term=10.0, noise_normalization=30.0)

        assert evidences == -0.5 * (3.0 + 6.0 + 9.0 - 10.0 + 30.0)