from auto_lens.imaging import hyper_image

import numpy as np

class TestHyperImage(object):

    class TestContributionMaps(object):

        def test__1_galaxy__model_image_all_1s__factor_is_0__contributions_all_1s(self):

            galaxy_image = np.ones((3))

            hyp = hyper_image.HyperImage(contribution_factors=(0.0,))
            contributions = hyp.compute_galaxy_contributions(model_image=galaxy_image, galaxy_image=galaxy_image,
                                                             galaxy_index=0, minimum_value=0.0)

            assert (contributions == np.ones((3))).all()

        def test__different_values__factor_is_1__contributions_are_value_divied_by_factor_and_max(self):

            galaxy_image = np.array([0.5, 1.0, 1.5])

            hyp = hyper_image.HyperImage(contribution_factors=(1.0,))
            contributions = hyp.compute_galaxy_contributions(model_image=galaxy_image, galaxy_image=galaxy_image,
                                                             galaxy_index=0, minimum_value=0.0)

            assert (contributions == np.array([(0.5/1.5)/(1.5/2.5), (1.0/2.0)/(1.5/2.5), 1.0])).all()

        def test__different_values__threshold_is_1_minimum_threshold_included__wipes_value_to_0(self):

            galaxy_image = np.array([0.5, 1.0, 1.5])

            hyp = hyper_image.HyperImage(contribution_factors=(1.0,))
            contributions = hyp.compute_galaxy_contributions(model_image=galaxy_image, galaxy_image=galaxy_image,
                                                             galaxy_index=0, minimum_value=0.6)

            assert (contributions == np.array([0.0, (1.0/2.0)/(1.5/2.5), 1.0])).all()

        def test__2_galaxies__returns_2_contribution_maps(self):

            # These both give 1.0/2.0 before division by max -> both scale to 1.0
            galaxy_image_0 = np.array([0.5, 1.0])
            galaxy_image_1 = np.array([0.5, 1.0]) # First term gives 0.5/2.0, second terms 1.0 / 3.0
            galaxy_images = [galaxy_image_0, galaxy_image_1]

            # Results:
            # galaxy_image_0[0] = (0.5)/(1.0+0.0) = 0.5
            # galaxy_image_0[1] = (1.0)/(2.0+0.0) = 0.5
            # After diving by max (2.0), both go to 1.0

            # galaxy_image_1[0] = (0.5)/(1.0+1.0) = (0.5/2.0) = 0.25
            # galaxy_image_1[1] = (1.0)/(2.0+1.0) = (1.0/3.0) = 0.333
            # After diving by max -> [0.25/0.333, 0.333/0.333] = [0.75, 1.0]

            hyp = hyper_image.HyperImage(contribution_factors=(0.0, 1.0))
            contributions = hyp.compute_all_galaxy_contributions(galaxy_images=galaxy_images, minimum_values=[0.0, 0.0])

            assert (contributions[0] == np.array([1.0, 1.0])).all()
            assert (contributions[1] == np.array([0.75, 1.0])).all()

    class TestScaledNoise(object):

        def test__1_galaxy__contribution_all_1s__no_background_scale__noise_factor_2_baseline_noise_adds_double(self):

            baseline_noise = np.array([1.0, 2.0, 3.0])
            background_noise = np.ones((3))
            galaxy_contributions = [np.ones((3))]

            hyp = hyper_image.HyperImage(contribution_factors=(0.0,), background_noise_scale=0.0,
                                         noise_factors=(2.0,), noise_powers=(1.0,))

            scaled_noise = hyp.compute_scaled_noise(grid_baseline_noise=baseline_noise,
                                                    grid_background_noise=background_noise,
                                                    galaxy_contributions=galaxy_contributions)

            assert (scaled_noise == np.array([3.0, 6.0, 9.0])).all()

        def test__same_as_above_but_contributions_vary(self):

            baseline_noise = np.array([1.0, 2.0, 3.0])
            background_noise = np.ones((3))
            galaxy_contributions = [np.array([0.0, 0.5, 1.0])]

            hyp = hyper_image.HyperImage(contribution_factors=(0.0,), background_noise_scale=0.0,
                                         noise_factors=(2.0, ), noise_powers=(1.0, ))

            scaled_noise = hyp.compute_scaled_noise(grid_baseline_noise=baseline_noise,
                                                    grid_background_noise=background_noise,
                                                    galaxy_contributions=galaxy_contributions)

            assert (scaled_noise == np.array([1.0, 4.0, 9.0])).all()

        def test__same_as_above_but_change_noise_scale_terms(self):

            baseline_noise = np.array([1.0, 2.0, 3.0])
            background_noise = np.ones((3))
            galaxy_contributions = [np.array([0.0, 0.5, 1.0])]

            hyp = hyper_image.HyperImage(contribution_factors=(0.0,), background_noise_scale=0.0,
                                         noise_factors=(2.0,), noise_powers=(2.0, ))

            scaled_noise = hyp.compute_scaled_noise(grid_baseline_noise=baseline_noise,
                                                    grid_background_noise=background_noise,
                                                    galaxy_contributions=galaxy_contributions)

            assert (scaled_noise == np.array([1.0, 4.0, 21.0])).all()

        def test__same_as_1st_test_but_also_scale_background(self):

            baseline_noise = np.array([1.0, 2.0, 3.0])
            background_noise = np.ones((3))
            galaxy_contributions = [np.ones((3))]

            hyp = hyper_image.HyperImage(contribution_factors=(0.0,), background_noise_scale=2.0,
                                         noise_factors=(2.0,), noise_powers=(1.0,))

            scaled_noise = hyp.compute_scaled_noise(grid_baseline_noise=baseline_noise,
                                                    grid_background_noise=background_noise,
                                                    galaxy_contributions=galaxy_contributions)

            assert (scaled_noise == np.array([5.0, 8.0, 11.0])).all()

        def test__2_galaxies__no_background__scaled_noise_is_sum_of_each_galaxy_component(self):

            baseline_noise = np.array([1.0, 2.0, 3.0])
            background_noise = np.ones((3))
            galaxy_contributions = [np.ones((3)), np.array([0.0, 0.5, 1.0])]

            hyp = hyper_image.HyperImage(contribution_factors=(0.0, 0.0), background_noise_scale=0.0,
                                         noise_factors=(2.0, 2.0), noise_powers=(1.0, 2.0))

            scaled_noise = hyp.compute_scaled_noise(grid_baseline_noise=baseline_noise,
                                                    grid_background_noise=background_noise,
                                                    galaxy_contributions=galaxy_contributions)

            # scaled_noise_0 == np.array([2.0, 4.0, 6.0]))
            # scaled_noise_1 == np.array([0.0, 2.0, 18.0]))

            assert (scaled_noise == np.array([3.0, 8.0, 27.0])).all()
