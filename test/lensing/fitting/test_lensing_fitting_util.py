import numpy as np
import pytest
from astropy import cosmology as cosmo

from autolens.data.array import mask as msk
from autolens.data.array import grids
from autolens.data.imaging import convolution
from autolens.model.profiles import light_profiles as lp
from autolens.lensing.fitting import lensing_fitting_util
from autolens.lensing import plane as pl
from autolens.lensing import ray_tracing
from autolens.model.galaxy import galaxy as g
from test.mock.mock_galaxy import MockHyperGalaxy

@pytest.fixture(name='mask')
def make_mask():
    return msk.Mask(array=np.array([[True, True, True, True],
                                     [True, False, False, True],
                                     [True, False, False, True],
                                     [True, True, True, True]]), pixel_scale=1.0)

@pytest.fixture(name='blurring_mask')
def make_blurring_mask():
    return msk.Mask(array=np.array([[False, False, False, False],
                                     [False, True, True, False],
                                     [False, True, True, False],
                                     [False, False, False, False]]), pixel_scale=1.0)

@pytest.fixture(name='convolver_no_blur')
def make_convolver_no_blur(mask, blurring_mask):

    psf = np.array([[0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0]])

    return convolution.ConvolverImage(mask=mask, blurring_mask=blurring_mask, psf=psf)

@pytest.fixture(name='convolver_blur')
def make_convolver_blur(mask, blurring_mask):

    psf = np.array([[1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]])

    return convolution.ConvolverImage(mask=mask, blurring_mask=blurring_mask, psf=psf)


class TestBlurImages:

    def test__2x2_image_all_1s__3x3__psf_central_1__no_blurring(self, convolver_no_blur):

        image_ = np.array([1.0, 1.0, 1.0, 1.0])
        blurring_image_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        blurred_image_ = lensing_fitting_util.blurred_image_1d_from_1d_unblurred_and_blurring_images(unblurred_image_1d=image_,
                                                                                                     blurring_image_1d=blurring_image_, convolver=convolver_no_blur)

        assert (blurred_image_ == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__2x2_image_all_1s__3x3_psf_all_1s__image_blurs_to_4s(self, convolver_blur):

        image_ = np.array([1.0, 1.0, 1.0, 1.0])
        blurring_image_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        blurred_image_ = lensing_fitting_util.blurred_image_1d_from_1d_unblurred_and_blurring_images(unblurred_image_1d=image_,
                                                                                                     blurring_image_1d=blurring_image_,
                                                                                                     convolver=convolver_blur)
        assert (blurred_image_ == np.array([4.0, 4.0, 4.0, 4.0])).all()


class TestLensingConvolutionFitter:

    def test__2x2_unblurred_image_all_1s__3x3_psf_no_blurring__image_blurs_to_itself(self, mask, convolver_no_blur):

        unblurred_image_1d = np.array([1.0, 1.0, 1.0, 1.0])
        blurring_image_1d = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        regular_grid = grids.RegularGrid.from_mask(mask=mask)

        blurred_image = lensing_fitting_util.blurred_image_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=unblurred_image_1d, blurring_image_1d=blurring_image_1d, convolver=convolver_no_blur,
            map_to_scaled_array=regular_grid.map_to_2d)

        assert (blurred_image == np.array([[0.0, 0.0, 0.0, 0.0],
                                           [0.0, 1.0, 1.0, 0.0],
                                           [0.0, 1.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0]])).all()

    def test__2x2_unblurred_image_all_1s__3x3_psf_all_1s__image_blurs_to_9s(self, mask, convolver_blur):

        unblurred_image_1d = np.array([1.0, 1.0, 1.0, 1.0])
        blurring_image_1d = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        regular_grid = grids.RegularGrid.from_mask(mask=mask)

        blurred_image = lensing_fitting_util.blurred_image_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=unblurred_image_1d, blurring_image_1d=blurring_image_1d, convolver=convolver_blur,
            map_to_scaled_array=regular_grid.map_to_2d)

        assert (blurred_image == np.array([[0.0, 0.0, 0.0, 0.0],
                                           [0.0, 9.0, 9.0, 0.0],
                                           [0.0, 9.0, 9.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0]])).all()

    def test__2x2_image__psf_is_non_symmetric_producing_l_shape(self, mask, blurring_mask):

        psf = np.array([[0.0, 3.0, 0.0],
                        [0.0, 2.0, 1.0],
                        [0.0, 0.0, 0.0]])

        convolver = convolution.ConvolverImage(mask=mask, blurring_mask=blurring_mask, psf=psf)

        unblurred_image_1d = np.array([1.0, 2.0, 3.0, 4.0])
        blurring_image_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

        regular_grid = grids.RegularGrid.from_mask(mask=mask)

        blurred_image = lensing_fitting_util.blurred_image_from_1d_unblurred_and_blurring_images(
            unblurred_image_1d=unblurred_image_1d, blurring_image_1d=blurring_image_1d, convolver=convolver,
            map_to_scaled_array=regular_grid.map_to_2d)

        # Manually compute result of convolution, which is each central value *2.0 plus its 2 appropriate neighbors

        blurred_image_manual_0 = 2.0 * unblurred_image_1d[0] + 3.0 * unblurred_image_1d[2] + blurring_image_1d[4]
        blurred_image_manual_1 = 2.0 * unblurred_image_1d[1] + 3.0 * unblurred_image_1d[3] + unblurred_image_1d[0]
        blurred_image_manual_2 = 2.0 * unblurred_image_1d[2] + 3.0 * blurring_image_1d[9] + blurring_image_1d[6]
        blurred_image_manual_3 = 2.0 * unblurred_image_1d[3] + 3.0 * blurring_image_1d[10] + unblurred_image_1d[2]

        assert blurred_image_manual_0 == pytest.approx(blurred_image[1, 1], 1e-6)
        assert blurred_image_manual_1 == pytest.approx(blurred_image[1, 2], 1e-6)
        assert blurred_image_manual_2 == pytest.approx(blurred_image[2, 1], 1e-6)
        assert blurred_image_manual_3 == pytest.approx(blurred_image[2, 2], 1e-6)


class TestInversionEvidence:

    def test__simple_values(self):

        likelihood_with_regularization_terms = \
            lensing_fitting_util.likelihood_with_regularization_from_chi_squared_term_regularization_and_noise_term(
                chi_squared_term=3.0, regularization_term=6.0, noise_term=2.0)

        assert likelihood_with_regularization_terms == -0.5 * (3.0 + 6.0 + 2.0)

        evidences = lensing_fitting_util.evidence_from_reconstruction_terms(chi_squared_term=3.0, regularization_term=6.0,
                                                                    log_covariance_regularization_term=9.0,
                                                                    log_regularization_term=10.0, noise_term=30.0)

        assert evidences == -0.5 * (3.0 + 6.0 + 9.0 - 10.0 + 30.0)


class TestBlurredImageOfPlanes:

    def test__blurred_image_of_planes__real_tracer__2x2_image__psf_is_non_symmetric_producing_l_shape(self, mask,
                                                                                                     convolver_blur):

        data_grids = grids.DataGridStack.grids_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=1,
                                                                                     psf_shape=(3,3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0))

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g1],
                                                     image_plane_grids=[data_grids])

        blurred_lens_image = convolver_blur.convolve_image(image_array=tracer.image_plane.image_plane_image_1d[0],
                                                           blurring_array=tracer.image_plane.image_plane_blurring_image_1d[0])
        blurred_lens_image = data_grids.regular.scaled_array_from_array_1d(array_1d=blurred_lens_image)

        blurred_source_image = convolver_blur.convolve_image(image_array=tracer.source_plane.image_plane_image_1d[0],
                                                             blurring_array=tracer.source_plane.image_plane_blurring_image_1d[0])
        blurred_source_image = data_grids.regular.scaled_array_from_array_1d(array_1d=blurred_source_image)
        
        blurred_image_of_planes = lensing_fitting_util.blurred_image_of_planes_from_tracer_and_convolver(tracer=tracer,
                                        convolver_image=convolver_blur, map_to_scaled_array=data_grids.regular.map_to_2d)

        assert (blurred_image_of_planes[0] == blurred_lens_image).all()
        assert (blurred_image_of_planes[1] == blurred_source_image).all()

    def test__same_as_above_but_multi_tracer(self, mask, convolver_blur):

        data_grids = grids.DataGridStack.grids_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=1,
                                                                                     psf_shape=(3,3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0), redshift=0.1)
        g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=2.0), redshift=0.2)
        g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=3.0), redshift=0.3)

        tracer = ray_tracing.TracerMulti(galaxies=[g0, g1, g2], image_plane_grids=[data_grids],
                                         cosmology=cosmo.Planck15)

        blurred_plane_image_0 = convolver_blur.convolve_image(tracer.planes[0].image_plane_image_1d[0],
                                                              tracer.planes[0].image_plane_blurring_image_1d[0])

        blurred_plane_image_1 = convolver_blur.convolve_image(tracer.planes[1].image_plane_image_1d[0],
                                                              tracer.planes[1].image_plane_blurring_image_1d[0])

        blurred_plane_image_2 = convolver_blur.convolve_image(tracer.planes[2].image_plane_image_1d[0],
                                                              tracer.planes[2].image_plane_blurring_image_1d[0])

        blurred_plane_image_0 = data_grids.regular.scaled_array_from_array_1d(blurred_plane_image_0)
        blurred_plane_image_1 = data_grids.regular.scaled_array_from_array_1d(blurred_plane_image_1)
        blurred_plane_image_2 = data_grids.regular.scaled_array_from_array_1d(blurred_plane_image_2)

        blurred_image_of_planes = lensing_fitting_util.blurred_image_of_planes_from_tracer_and_convolver(tracer=tracer,
                                  convolver_image=convolver_blur, map_to_scaled_array=data_grids.regular.map_to_2d)

        assert (blurred_image_of_planes[0] == blurred_plane_image_0).all()
        assert (blurred_image_of_planes[1] == blurred_plane_image_1).all()
        assert (blurred_image_of_planes[2] == blurred_plane_image_2).all()

    def test__blurred_images_of_planes__if_galaxy_has_no_light_profile__replace_with_none(self, mask, convolver_blur):

        data_grids = grids.DataGridStack.grids_from_mask_sub_grid_size_and_psf_shape(mask=mask, sub_grid_size=1,
                                                                                     psf_shape=(3,3))

        g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=1.0))
        g0_image = pl.intensities_from_grid(grid=data_grids.sub, galaxies=[g0])
        g0_blurring_image = pl.intensities_from_grid(grid=data_grids.blurring, galaxies=[g0])
        g0_blurred_image_1d = convolver_blur.convolve_image(image_array=g0_image, blurring_array=g0_blurring_image)
        g0_blurred_image = data_grids.regular.scaled_array_from_array_1d(array_1d=g0_blurred_image_1d)

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g0],
                                                     image_plane_grids=[data_grids])

        blurred_image_of_planes = lensing_fitting_util.blurred_image_of_planes_from_tracer_and_convolver(tracer=tracer,
                                  convolver_image=convolver_blur, map_to_scaled_array=data_grids.regular.map_to_2d)

        assert (blurred_image_of_planes[0] == g0_blurred_image).all()
        assert (blurred_image_of_planes[1] == g0_blurred_image).all()

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0], source_galaxies=[g.Galaxy()],
                                                     image_plane_grids=[data_grids])

        blurred_image_of_planes = lensing_fitting_util.blurred_image_of_planes_from_tracer_and_convolver(tracer=tracer,
                                  convolver_image=convolver_blur, map_to_scaled_array=data_grids.regular.map_to_2d)

        assert (blurred_image_of_planes[0] == g0_blurred_image).all()
        assert blurred_image_of_planes[1] == None

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[g0],
                                                     image_plane_grids=[data_grids])

        blurred_image_of_planes = lensing_fitting_util.blurred_image_of_planes_from_tracer_and_convolver(tracer=tracer,
                                  convolver_image=convolver_blur, map_to_scaled_array=data_grids.regular.map_to_2d)

        assert blurred_image_of_planes[0] == None
        assert (blurred_image_of_planes[1] == g0_blurred_image).all()

        tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g.Galaxy()], source_galaxies=[g.Galaxy()],
                                                     image_plane_grids=[data_grids])

        blurred_image_of_planes = lensing_fitting_util.blurred_image_of_planes_from_tracer_and_convolver(tracer=tracer,
                                  convolver_image=convolver_blur, map_to_scaled_array=data_grids.regular.map_to_2d)

        assert blurred_image_of_planes[0] == None
        assert blurred_image_of_planes[1] == None


class TestContributionsFromHypers:

    def test__x1_hyper_galaxy__model_is_galaxy_image__contributions_all_1(self):

        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image_ = np.array([1.0, 1.0, 1.0])

        hyper_galaxy_images_ = [np.array([1.0, 1.0, 1.0])]

        minimum_values = [0.0]

        contributions = lensing_fitting_util.contributions_from_hyper_images_and_galaxies(
            hyper_model_image=hyper_model_image_, hyper_galaxy_images=hyper_galaxy_images_, hyper_galaxies=hyper_galaxies,
            minimum_values=minimum_values)

        assert (contributions[0] == np.array([1.0, 1.0, 1.0])).all()

    def test__x1_hyper_galaxy__model_and_galaxy_image_different_contributions_change(self):

        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image_ = np.array([0.5, 1.0, 1.5])

        hyper_galaxy_images_ = [np.array([0.5, 1.0, 1.5])]

        minimum_values = [0.6]

        contributions = lensing_fitting_util.contributions_from_hyper_images_and_galaxies(
            hyper_model_image=hyper_model_image_, hyper_galaxy_images=hyper_galaxy_images_, hyper_galaxies=hyper_galaxies,
            minimum_values=minimum_values)

        assert (contributions[0] == np.array([0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0])).all()

    def test__x2_hyper_galaxy__model_and_galaxy_image_different_contributions_change(self):

        hyper_galaxies = [MockHyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image_ = np.array([0.5, 1.0, 1.5])

        hyper_galaxy_images_ = [np.array([0.5, 1.0, 1.5]), np.array([0.5, 1.0, 1.5])]

        minimum_values = [0.5, 0.6]

        contributions = lensing_fitting_util.contributions_from_hyper_images_and_galaxies(
            hyper_model_image=hyper_model_image_, hyper_galaxy_images=hyper_galaxy_images_, hyper_galaxies=hyper_galaxies,
            minimum_values=minimum_values)

        assert (contributions[0] == np.array([1.0, 1.0, 1.0])).all()
        assert (contributions[1] == np.array([0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0])).all()

    def test__x2_hyper_galaxy__same_as_above_use_real_hyper_galaxy(self):

        hyper_galaxies = [g.HyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
                          g.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]

        hyper_model_image_ = np.array([0.5, 1.0, 1.5])

        hyper_galaxy_images_ = [np.array([0.5, 1.0, 1.5]), np.array([0.5, 1.0, 1.5])]

        minimum_values = [0.5, 0.6]

        contributions = lensing_fitting_util.contributions_from_hyper_images_and_galaxies(
            hyper_model_image=hyper_model_image_, hyper_galaxy_images=hyper_galaxy_images_, hyper_galaxies=hyper_galaxies,
            minimum_values=minimum_values)

        assert (contributions[0] == np.array([1.0, 1.0, 1.0])).all()
        assert (contributions[1] == np.array([0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0])).all()

    # def test__same_as_above__contributions_from_fitting_hyper_images(self, image, masks):
    # 
    #     hyper_galaxies = [g.HyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
    #                       g.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]
    # 
    #     hyper_model_image = np.array([[0.5, 1.0, 1.5]])
    # 
    #     hyper_galaxy_images = [np.array([[0.5, 1.0, 1.5]]), np.array([[0.5, 1.0, 1.5]])]
    # 
    #     minimum_values = [0.5, 0.6]
    # 
    #     fitting_hyper_image = fit_data.FitDataHyper(datas=image, masks=masks, hyper_model_image=hyper_model_image,
    #                                                 hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=minimum_values)
    # 
    #     contributions = lensing_lensing_fitting_util.contributions_from_fitting_hyper_images_and_hyper_galaxies(
    #         fitting_hyper_images=[fitting_hyper_image], hyper_galaxies=hyper_galaxies)
    # 
    #     assert (contributions[0][0] == np.array([[1.0, 1.0, 1.0]])).all()
    #     assert (contributions[0][1] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()
    # 
    # def test__same_as_above__x2_images(self, image, masks):
    # 
    #     hyper_galaxies = [g.HyperGalaxy(contribution_factor=0.0, noise_factor=0.0, noise_power=1.0),
    #                       g.HyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]
    # 
    #     hyper_model_image = np.array([[0.5, 1.0, 1.5]])
    #     hyper_galaxy_images = [np.array([[0.5, 1.0, 1.5]]), np.array([[0.5, 1.0, 1.5]])]
    #     minimum_values = [0.5, 0.6]
    # 
    #     fitting_hyper_image_0 = fit_data.FitDataHyper(datas=image, masks=masks, hyper_model_image=hyper_model_image,
    #                                                   hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=minimum_values)
    # 
    #     fitting_hyper_image_1 = fit_data.FitDataHyper(datas=image, masks=masks, hyper_model_image=hyper_model_image,
    #                                                   hyper_galaxy_images=hyper_galaxy_images, hyper_minimum_values=minimum_values)
    # 
    #     contributions = lensing_lensing_fitting_util.contributions_from_fitting_hyper_images_and_hyper_galaxies(
    #         fitting_hyper_images=[fitting_hyper_image_0, fitting_hyper_image_1], hyper_galaxies=hyper_galaxies)
    # 
    #     assert (contributions[0][0] == np.array([[1.0, 1.0, 1.0]])).all()
    #     assert (contributions[0][1] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()
    #     assert (contributions[1][0] == np.array([[1.0, 1.0, 1.0]])).all()
    #     assert (contributions[1][1] == np.array([[0.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])).all()


class TestScaledNoiseFromContributions:

    def test__x1_hyper_galaxy__noise_factor_is_0__scaled_noise_is_input_noise(self):

        contributions_ = [np.array([1.0, 1.0, 2.0])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=0.0, noise_power=1.0)]
        noise_map_ = np.array([1.0, 1.0, 1.0])

        scaled_noise_map_ = lensing_fitting_util.scaled_noise_map_from_hyper_galaxies_and_contributions(
            contributions=contributions_, hyper_galaxies=hyper_galaxies, noise_map=noise_map_)

        assert (scaled_noise_map_ == noise_map_).all()

    def test__x1_hyper_galaxy__noise_factor_and_power_are_1__scaled_noise_added_to_input_noise(self):

        contributions_ = [np.array([1.0, 1.0, 0.5])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=1.0)]
        noise_map_ = np.array([1.0, 1.0, 1.0])

        scaled_noise_map_ = lensing_fitting_util.scaled_noise_map_from_hyper_galaxies_and_contributions(
            contributions=contributions_, hyper_galaxies=hyper_galaxies, noise_map=noise_map_)

        assert (scaled_noise_map_ == np.array([2.0, 2.0, 1.5])).all()

    def test__x1_hyper_galaxy__noise_factor_1_and_power_is_2__scaled_noise_added_to_input_noise(self):

        contributions_ = [np.array([1.0, 1.0, 0.5])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0)]
        noise_map_ = np.array([1.0, 1.0, 1.0])

        scaled_noise_map_ = lensing_fitting_util.scaled_noise_map_from_hyper_galaxies_and_contributions(
            contributions=contributions_, hyper_galaxies=hyper_galaxies, noise_map=noise_map_)

        assert (scaled_noise_map_ == np.array([2.0, 2.0, 1.25])).all()

    def test__x2_hyper_galaxy__noise_factor_1_and_power_is_2__scaled_noise_added_to_input_noise(self):

        contributions_ = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
        hyper_galaxies = [MockHyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
                          MockHyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
        noise_map_ = np.array([1.0, 1.0, 1.0])

        scaled_noise_map_ = lensing_fitting_util.scaled_noise_map_from_hyper_galaxies_and_contributions(
            contributions=contributions_, hyper_galaxies=hyper_galaxies, noise_map=noise_map_)

        assert (scaled_noise_map_ == np.array([2.5, 2.5, 1.75])).all()

    def test__x2_hyper_galaxy__same_as_above_but_use_real_hyper_galaxy(self):

        contributions_ = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
        hyper_galaxies = [g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
                          g.HyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
        noise_map_ = np.array([1.0, 1.0, 1.0])

        scaled_noise_map_ = lensing_fitting_util.scaled_noise_map_from_hyper_galaxies_and_contributions(
            contributions=contributions_, hyper_galaxies=hyper_galaxies, noise_map=noise_map_)

        assert (scaled_noise_map_ == np.array([2.5, 2.5, 1.75])).all()

    # def test__same_as_above_but_using_fiting_hyper_image(self, image, masks):
    # 
    #     fitting_hyper_image = fit_data.FitDataHyper(datas=image, masks=masks, hyper_model_image=None,
    #                                                 hyper_galaxy_images=None, hyper_minimum_values=None)
    # 
    #     contributions = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
    #     hyper_galaxies = [g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
    #                       g.HyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
    #     scaled_noises = lensing_lensing_fitting_util.scaled_noise_maps_from_fitting_hyper_images_contributions_and_hyper_galaxies(
    #         fitting_hyper_images=[fitting_hyper_image], contributions_=[contributions], hyper_galaxies=hyper_galaxies)
    # 
    #     assert (scaled_noises[0] == np.array([2.5, 2.5, 1.75])).all()
    # 
    # def test__same_as_above_but_using_x2_fiting_hyper_image(self, image, masks):
    # 
    #     fitting_hyper_image_0 = fit_data.FitDataHyper(datas=image, masks=masks, hyper_model_image=None,
    #                                                   hyper_galaxy_images=None, hyper_minimum_values=None)
    #     fitting_hyper_image_1 = fit_data.FitDataHyper(datas=image, masks=masks, hyper_model_image=None,
    #                                                   hyper_galaxy_images=None, hyper_minimum_values=None)
    # 
    #     contributions_0 = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
    #     contributions_1 = [np.array([1.0, 1.0, 0.5]), np.array([0.25, 0.25, 0.25])]
    #     hyper_galaxies = [g.HyperGalaxy(contribution_factor=1.0, noise_factor=1.0, noise_power=2.0),
    #                       g.HyperGalaxy(contribution_factor=1.0, noise_factor=2.0, noise_power=1.0)]
    #     scaled_noises = lensing_lensing_fitting_util.scaled_noise_maps_from_fitting_hyper_images_contributions_and_hyper_galaxies(
    #         fitting_hyper_images=[fitting_hyper_image_0, fitting_hyper_image_1],
    #         contributions_=[contributions_0, contributions_1], hyper_galaxies=hyper_galaxies)
    # 
    #     assert (scaled_noises[0] == np.array([2.5, 2.5, 1.75])).all()
    #     assert (scaled_noises[1] == np.array([2.5, 2.5, 1.75])).all()


# class TestUnmaskedModelImages:
#
#     def test___3x3_padded_image__no_psf_blurring__produces_padded_image(self):
#
#         psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
#                                          [0.0, 1.0, 0.0],
#                                          [0.0, 0.0, 0.0]])), pixel_scale=1.0)
#         image = im.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_maps=np.ones((3, 3)))
#
#         masks = msk.Mask(array=np.array([[True, True, True],
#                                        [True, False, True],
#                                        [True, True, True]]), pixel_scale=1.0)
#
#         fitting_image = fit_data.FitData(image, masks, sub_grid_size=1)
#
#         padded_model_image = lensing_lensing_lensing_fitting_util.unmasked_blurred_images_from_fitting_images(fitting_images=[fitting_image],
#                                                                                  unmasked_images_=[np.ones(25)])
#
#         assert (padded_model_image[0] == np.ones((3,3))).all()
#
#     def test___3x3_padded_image__simple_psf_blurring__produces_padded_image(self):
#
#         psf = im.PSF(array=(np.array([[0.0, 0.0, 0.0],
#                                          [0.0, 1.0, 2.0],
#                                          [0.0, 0.0, 0.0]])), pixel_scale=1.0)
#         image = im.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_maps=np.ones((3, 3)))
#
#         masks = msk.Mask(array=np.array([[True, True, True],
#                                        [True, False, True],
#                                        [True, True, True]]), pixel_scale=1.0)
#
#         fitting_image = fit_data.FitData(image, masks, sub_grid_size=1)
#
#         padded_model_image = lensing_lensing_lensing_fitting_util.unmasked_blurred_images_from_fitting_images(fitting_images=[fitting_image],
#                                                                                  unmasked_images_=[np.ones(25)])
#
#         assert (padded_model_image == 3.0*np.ones((3, 3))).all()
#
#     def test___3x3_padded_image__asymmetric_psf_blurring__produces_padded_image(self):
#
#         psf = im.PSF(array=(np.array([[0.0, 3.0, 0.0],
#                                          [0.0, 1.0, 2.0],
#                                          [0.0, 0.0, 0.0]])), pixel_scale=1.0)
#         image = im.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_maps=np.ones((3, 3)))
#
#         masks = msk.Mask(array=np.array([[True, True, True],
#                                        [True, False, True],
#                                        [True, True, True]]), pixel_scale=1.0)
#
#         fitting_image = fit_data.FitData(image, masks, sub_grid_size=1)
#
#         _unmasked_image = np.zeros(25)
#         _unmasked_image[12] = 1.0
#
#         padded_model_image = lensing_lensing_lensing_fitting_util.unmasked_blurred_images_from_fitting_images(fitting_images=[fitting_image],
#                                                                                  unmasked_images_=[_unmasked_image])
#
#         assert (padded_model_image == np.array([[0.0, 3.0, 0.0],
#                                                 [0.0, 1.0, 2.0],
#                                                 [0.0, 0.0, 0.0]])).all()


# class TestUnmaskedModelImages:
#
#     def test___of_galaxies__x1_galaxy__3x3_padded_image__no_psf_blurring(self, galaxy_light):
#
#         psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
#                                          [0.0, 1.0, 0.0],
#                                          [0.0, 0.0, 0.0]])), pixel_scale=1.0)
#         im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_maps=np.ones((3, 3)))
#
#         ma = masks.Mask(array=np.array([[True, True, True],
#                                        [True, False, True],
#                                        [True, True, True]]), pixel_scale=1.0)
#         li = lensing_image.LensingImage(im, ma, sub_grid_size=1)
#
#         tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=[li.padded_grids])
#
#         manual_model_image_0 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(tracer.image_plane_images_[0])
#         manual_model_image_0 = psf.convolve(manual_model_image_0)
#
#         padded_model_images = lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(
#             lensing_image=li, tracer=tracer, image_index=0)
#
#         assert (manual_model_image_0[1:4, 1:4] == padded_model_images[0][0]).all()
#
#     def test___of_galaxies__x1_galaxy__3x3_padded_image__asymetric_psf_blurring(self, galaxy_light):
#
#         psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
#                                          [0.0, 1.0, 0.0],
#                                          [0.0, 0.0, 0.0]])), pixel_scale=1.0)
#         im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_maps=np.ones((3, 3)))
#
#         ma = masks.Mask(array=np.array([[True, True, True],
#                                        [True, False, True],
#                                        [True, True, True]]), pixel_scale=1.0)
#         li = lensing_image.LensingImage(im, ma, sub_grid_size=1)
#
#         tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=[li.padded_grids])
#
#         manual_model_image_0 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(tracer.image_plane_images_[0])
#         manual_model_image_0 = psf.convolve(manual_model_image_0)
#
#         padded_model_images = lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(
#             lensing_image=li, tracer=tracer, image_index=0)
#
#         assert (manual_model_image_0[1:4, 1:4] == padded_model_images[0][0]).all()
#
#     def test___of_galaxies__x2_galaxies__3x3_padded_image__asymetric_psf_blurring(self):
#         psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
#                                          [0.0, 1.0, 0.0],
#                                          [0.0, 0.0, 0.0]])), pixel_scale=1.0)
#         im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_maps=np.ones((3, 3)))
#
#         ma = masks.Mask(array=np.array([[True, True, True],
#                                        [True, False, True],
#                                        [True, True, True]]), pixel_scale=1.0)
#         li = lensing_image.LensingImage(im, ma, sub_grid_size=1)
#
#         g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
#         g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))
#
#         tracer = ray_tracing.TracerImagePlane(lens_galaxies=[g0, g1], image_plane_grids=[li.padded_grids])
#
#         manual_model_image_0 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.image_plane.image_plane_images_of_galaxies_[0][0])
#         manual_model_image_0 = psf.convolve(manual_model_image_0)
#
#         manual_model_image_1 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.image_plane.image_plane_images_of_galaxies_[0][1])
#         manual_model_image_1 = psf.convolve(manual_model_image_1)
#
#         padded_model_images = lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(
#             lensing_image=li, tracer=tracer, image_index=0)
#
#         assert (manual_model_image_0[1:4, 1:4] == padded_model_images[0][0]).all()
#         assert (manual_model_image_1[1:4, 1:4] == padded_model_images[0][1]).all()
#
#     def test___same_as_above_but_image_and_souce_plane(self):
#
#         psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
#                                          [0.0, 1.0, 0.0],
#                                          [0.0, 0.0, 0.0]])), pixel_scale=1.0)
#         im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_maps=np.ones((3, 3)))
#
#         ma = masks.Mask(array=np.array([[True, True, True],
#                                        [True, False, True],
#                                        [True, True, True]]), pixel_scale=1.0)
#         li = lensing_image.LensingImage(im, ma, sub_grid_size=1)
#
#         g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1))
#         g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))
#         g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.3))
#         g3 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.4))
#
#         tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
#                                                      image_plane_grids=[li.padded_grids])
#
#         manual_model_image_0 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.image_plane.image_plane_images_of_galaxies_[0][0])
#         manual_model_image_0 = psf.convolve(manual_model_image_0)
#
#         manual_model_image_1 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.image_plane.image_plane_images_of_galaxies_[0][1])
#         manual_model_image_1 = psf.convolve(manual_model_image_1)
#
#
#         manual_model_image_2 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.source_plane.image_plane_images_of_galaxies_[0][0])
#         manual_model_image_2 = psf.convolve(manual_model_image_2)
#
#         manual_model_image_3 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.source_plane.image_plane_images_of_galaxies_[0][1])
#         manual_model_image_3 = psf.convolve(manual_model_image_3)
#
#         padded_model_images = lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(
#             lensing_image=li, tracer=tracer, image_index=0)
#
#         assert (manual_model_image_0[1:4, 1:4] == padded_model_images[0][0]).all()
#         assert (manual_model_image_1[1:4, 1:4] == padded_model_images[0][1]).all()
#         assert (manual_model_image_2[1:4, 1:4] == padded_model_images[1][0]).all()
#         assert (manual_model_image_3[1:4, 1:4] == padded_model_images[1][1]).all()
#
#     def test___same_as_above_but_x2_images(self):
#
#         psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
#                                          [0.0, 1.0, 0.0],
#                                          [0.0, 0.0, 0.0]])), pixel_scale=1.0)
#
#         ma = masks.Mask(array=np.array([[True, True, True],
#                                        [True, False, True],
#                                        [True, True, True]]), pixel_scale=1.0)
#
#         im_0 = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_maps=np.ones((3, 3)))
#         li_0 = lensing_image.LensingImage(im_0, ma, sub_grid_size=1)
#
#         im_1 = image.Image(array=2.0 * np.ones((3, 3)), pixel_scale=2.0, psf=psf, noise_maps=np.ones((3, 3)))
#         li_1 = lensing_image.LensingImage(im_1, ma, sub_grid_size=2)
#
#         g0 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.1), centre=(0.1, 0.1))
#         g1 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.2))
#         g2 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.3))
#         g3 = g.Galaxy(light_profile=lp.EllipticalSersic(intensity=0.4))
#
#         tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[g0, g1], source_galaxies=[g2, g3],
#                                                      image_plane_grids=[li_0.padded_grids, li_1.padded_grids])
#
#         manual_model_image_0_of_image_0 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.image_plane.image_plane_images_of_galaxies_[0][0])
#         manual_model_image_0_of_image_0 = psf.convolve(manual_model_image_0_of_image_0)
#
#         manual_model_image_1_of_image_0 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.image_plane.image_plane_images_of_galaxies_[0][1])
#         manual_model_image_1_of_image_0 = psf.convolve(manual_model_image_1_of_image_0)
#
#         manual_model_image_2_of_image_0 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.source_plane.image_plane_images_of_galaxies_[0][0])
#         manual_model_image_2_of_image_0 = psf.convolve(manual_model_image_2_of_image_0)
#
#         manual_model_image_3_of_image_0 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.source_plane.image_plane_images_of_galaxies_[0][1])
#         manual_model_image_3_of_image_0 = psf.convolve(manual_model_image_3_of_image_0)
#
#         manual_model_image_0_of_image_1 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.image_plane.image_plane_images_of_galaxies_[1][0])
#         manual_model_image_0_of_image_1 = psf.convolve(manual_model_image_0_of_image_1)
#
#         manual_model_image_1_of_image_1 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.image_plane.image_plane_images_of_galaxies_[1][1])
#         manual_model_image_1_of_image_1 = psf.convolve(manual_model_image_1_of_image_1)
#
#         manual_model_image_2_of_image_1 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.source_plane.image_plane_images_of_galaxies_[1][0])
#         manual_model_image_2_of_image_1 = psf.convolve(manual_model_image_2_of_image_1)
#
#         manual_model_image_3_of_image_1 = tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(
#             tracer.source_plane.image_plane_images_of_galaxies_[1][1])
#         manual_model_image_3_of_image_1 = psf.convolve(manual_model_image_3_of_image_1)
#
#         padded_model_images = lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_images_and_tracer(
#             lensing_images=[li_0, li_1], tracer=tracer)
#
#         assert (manual_model_image_0_of_image_0[1:4, 1:4] == padded_model_images[0][0][0]).all()
#         assert (manual_model_image_1_of_image_0[1:4, 1:4] == padded_model_images[0][0][1]).all()
#         assert (manual_model_image_2_of_image_0[1:4, 1:4] == padded_model_images[0][1][0]).all()
#         assert (manual_model_image_3_of_image_0[1:4, 1:4] == padded_model_images[0][1][1]).all()
#
#         assert (manual_model_image_0_of_image_1[1:4, 1:4] == padded_model_images[1][0][0]).all()
#         assert (manual_model_image_1_of_image_1[1:4, 1:4] == padded_model_images[1][0][1]).all()
#         assert (manual_model_image_2_of_image_1[1:4, 1:4] == padded_model_images[1][1][0]).all()
#         assert (manual_model_image_3_of_image_1[1:4, 1:4] == padded_model_images[1][1][1]).all()
#
#     def test__properties_of_fit(self, galaxy_light):
#
#         psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
#                                          [0.0, 1.0, 0.0],
#                                          [0.0, 0.0, 0.0]])), pixel_scale=1.0)
#         im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_maps=np.ones((3, 3)))
#
#         ma = masks.Mask(array=np.array([[True, True, True],
#                                        [True, False, True],
#                                        [True, True, True]]), pixel_scale=1.0)
#         li = lensing_image.LensingImage(im, ma, sub_grid_size=1)
#
#         padded_tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=[li.padded_grids])
#
#         manual_model_image_0 = \
#             padded_tracer.image_plane.grid_stacks[0].regular.map_to_2d_keep_padded(padded_tracer.image_plane_images_[0])
#         manual_model_image_0 = psf.convolve(manual_model_image_0)
#
#         padded_model_images = lensing_fitting.unmasked_model_images_of_galaxies_from_lensing_image_and_tracer(
#             lensing_image=li, tracer=padded_tracer, image_index=0)
#
#         tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=[li.grid_stacks])
#
#         fit = lensing_fitting.LensingProfileFitter(lensing_images=[li], tracer=tracer, padded_tracer=padded_tracer)
#
#         assert (manual_model_image_0[1:4, 1:4] == fit.unmasked_model_profile_images[0]).all()
#         assert (padded_model_images[0][0] == fit.unmasked_model_profile_images_of_galaxies[0][0][0]).all()
#
#     def test__padded_tracer_is_none__returns_none(self, galaxy_light):
#
#         psf = image.PSF(array=(np.array([[0.0, 0.0, 0.0],
#                                          [0.0, 1.0, 0.0],
#                                          [0.0, 0.0, 0.0]])), pixel_scale=1.0)
#         im = image.Image(array=np.ones((3, 3)), pixel_scale=1.0, psf=psf, noise_maps=np.ones((3, 3)))
#
#         ma = masks.Mask(array=np.array([[True, True, True],
#                                        [True, False, True],
#                                        [True, True, True]]), pixel_scale=1.0)
#         li = lensing_image.LensingImage(im, ma, sub_grid_size=1)
#
#         tracer = ray_tracing.TracerImagePlane(lens_galaxies=[galaxy_light], image_plane_grids=[li.grid_stacks])
#
#         fit = lensing_fitting.LensingProfileFitter(lensing_images=[li], tracer=tracer, padded_tracer=None)
#
#         assert fit.unmasked_model_profile_images == None
#         assert fit.unmasked_model_profile_images_of_galaxies == None