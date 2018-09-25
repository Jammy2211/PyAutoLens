import numpy as np

from autolens.imaging import hyper_image


class TestHyperImage(object):
    class TestSkyScle(object):

        def test__scale_sky_in_image__increases_all_image_values(self):
            image = np.array([1.0, 2.0, 3.0])

            hyp = hyper_image.HyperImage(background_sky_scale=10.0)

            scaled_image = hyp.sky_scaled_image_from_image(image=image)

            assert (scaled_image == np.array([11.0, 12.0, 13.0])).all()

    class TestScaledNoise(object):

        def test__scaled_background_noise__adds_to_input_noise(self):
            noise = np.array([1.0, 2.0, 3.0])
            background_noise = np.ones((3))

            hyp = hyper_image.HyperImage(background_noise_scale=2.0)

            scaled_noise = hyp.scaled_noise_from_background_noise(noise=noise, background_noise=background_noise)

            assert (scaled_noise == np.array([3.0, 4.0, 5.0])).all()

        def test__variable_background_noise__adds_to_input_noise_different_for_each_value(self):
            noise = np.array([1.0, 2.0, 3.0])
            background_noise = np.array([5.0, 1.0, 3.0])

            hyp = hyper_image.HyperImage(background_noise_scale=3.0)

            scaled_noise = hyp.scaled_noise_from_background_noise(noise=noise, background_noise=background_noise)

            assert (scaled_noise == np.array([16.0, 5.0, 12.0])).all()
