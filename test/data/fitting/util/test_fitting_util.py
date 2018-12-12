import numpy as np
import pytest

from autolens.data.fitting.util import fitting_util


class TestResiduals:

    def test__model_matches_data__residuals_all_0s(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        model_data = np.array([10.0, 10.0, 10.0, 10.0])

        residuals = fitting_util.residuals_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)

        assert (residuals == np.array([0.0, 0.0, 0.0, 0.0])).all()

    def test__model_data_mismatch__no_masking__residuals_non_0(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting_util.residuals_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)

        assert (residuals == np.array([-1.0, 0.0, 1.0, 2.0])).all()

    def test__model_data_mismatch__mask_included__masked_residuals_set_to_0(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([True, False, False, True])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting_util.residuals_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)

        assert (residuals == np.array([0.0, 0.0, 1.0, 0.0])).all()


class TestChiSquareds:

    def test__model_mathces_data__chi_sq_all_0s(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([10.0, 10.0, 10.0, 10.0])

        residuals = fitting_util.residuals_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise_map(residuals=residuals, noise_map=noise_map)

        assert (chi_squareds == np.array([0.0, 0.0, 0.0, 0.0])).all()

    def test__model_data_mismatch__no_masking__chi_sq_non_0(self):
        
        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting_util.residuals_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise_map(residuals=residuals, noise_map=noise_map)

        assert (chi_squareds == np.array([(1.0 / 2.0)**2.0, 0.0, (1.0 / 2.0)**2.0, (2.0 / 2.0)**2.0])).all()

    def test__model_data_mismatch__mask_included__masked_chi_sqs_set_to_0(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([True, False, False, True])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting_util.residuals_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise_map(residuals=residuals, noise_map=noise_map)
        
        assert (chi_squareds == np.array([0.0, 0.0, (1.0 / 2.0)**2.0, 0.0])).all()


class TestLikelihood:

    def test__model_matches_data__noise_all_2s__lh_is_noise_term(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([10.0, 10.0, 10.0, 10.0])

        residuals = fitting_util.residuals_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise_map(residuals=residuals, noise_map=noise_map)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds=chi_squareds)
        noise_term = fitting_util.noise_term_from_mask_and_noise_map(mask=mask, noise_map=noise_map)
        likelihood = fitting_util.likelihood_from_chi_squared_term_and_noise_term(chi_squared_term=chi_squared_term,
                                                                                   noise_term=noise_term)

        chi_squared_term = 0.0
        noise_term = np.log(2.0 * np.pi * (2.0**2.0)) + np.log(2.0 * np.pi * (2.0**2.0)) + \
                     np.log(2.0 * np.pi * (2.0**2.0)) + np.log(2.0 * np.pi * (2.0**2.0))

        assert likelihood == -0.5 * (chi_squared_term + noise_term)

    def test__model_data_mismatch__no_masking__chi_squared_and_noise_term_are_lh(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        noise_map = np.array([2.0, 2.0, 2.0, 2.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting_util.residuals_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise_map(residuals=residuals, noise_map=noise_map)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds=chi_squareds)
        noise_term = fitting_util.noise_term_from_mask_and_noise_map(mask=mask, noise_map=noise_map)
        likelihood = fitting_util.likelihood_from_chi_squared_term_and_noise_term(chi_squared_term=chi_squared_term,
                                                                                   noise_term=noise_term)

        # chi squared = 0.25, 0, 0.25, 1.0
        # likelihood = -0.5*(0.25+0+0.25+1.0)

        chi_squared_term = ((1.0 / 2.0)**2.0) + 0.0 + ((1.0 / 2.0)**2.0) + ((2.0 / 2.0)**2.0)
        noise_term = np.log(2.0 * np.pi * (2.0**2.0)) + np.log(2.0 * np.pi * (2.0**2.0)) + \
                     np.log(2.0 * np.pi * (2.0**2.0)) + np.log(2.0 * np.pi * (2.0**2.0))

        assert likelihood == -0.5 * (chi_squared_term + noise_term)

    def test__same_as_above_but_different_noise_in_each_pixel(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([False, False, False, False])
        noise_map = np.array([1.0, 2.0, 3.0, 4.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting_util.residuals_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise_map(residuals=residuals, noise_map=noise_map)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds=chi_squareds)
        noise_term = fitting_util.noise_term_from_mask_and_noise_map(mask=mask, noise_map=noise_map)
        likelihood = fitting_util.likelihood_from_chi_squared_term_and_noise_term(chi_squared_term=chi_squared_term,
                                                                                  noise_term=noise_term)

        # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0

        chi_squared_term = 1.0 + (1.0 / (3.0**2.0)) + 0.25
        noise_term = np.log(2 * np.pi * (1.0**2.0)) + np.log(2 * np.pi * (2.0**2.0)) + \
                     np.log(2 * np.pi * (3.0**2.0)) + np.log(2 * np.pi * (4.0**2.0))

        assert likelihood == pytest.approx(-0.5 * (chi_squared_term + noise_term), 1e-4)

    def test__model_data_mismatch__mask_certain_pixels__lh_non_0(self):

        data = np.array([10.0, 10.0, 10.0, 10.0])
        mask = np.array([True, False, False, True])
        noise_map = np.array([1.0, 2.0, 3.0, 4.0])
        model_data = np.array([11.0, 10.0, 9.0, 8.0])

        residuals = fitting_util.residuals_from_data_mask_and_model_data(data=data, mask=mask, model_data=model_data)
        chi_squareds = fitting_util.chi_squareds_from_residuals_and_noise_map(residuals=residuals, noise_map=noise_map)
        chi_squared_term = fitting_util.chi_squared_term_from_chi_squareds(chi_squareds=chi_squareds)
        noise_term = fitting_util.noise_term_from_mask_and_noise_map(mask=mask, noise_map=noise_map)
        likelihood = fitting_util.likelihood_from_chi_squared_term_and_noise_term(chi_squared_term=chi_squared_term,
                                                                                  noise_term=noise_term)

        # chi squared = 0, 0.25, (0.25 and 1.0 are masked)

        chi_squared_term = 0.0 + (1.0 / 3.0)**2.0
        noise_term = np.log(2 * np.pi * (2.0**2.0)) + np.log(2 * np.pi * (3.0**2.0))

        assert likelihood == pytest.approx(-0.5 * (chi_squared_term + noise_term), 1e-4)