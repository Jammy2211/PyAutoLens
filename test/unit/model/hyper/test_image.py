import numpy as np

from autolens.model.hyper import image as hi

class TestHyperImage(object):

    class TestSkyScale(object):

        def test__scale_sky_in_image__increases_all_image_values(self):
            image = np.array([1.0, 2.0, 3.0])

            hyper_image = hi.HyperImage(background_sky_scale=10.0)

            scaled_image = hyper_image.sky_scaled_hyper_image_from_image(image=image)

            assert (scaled_image == np.array([11.0, 12.0, 13.0])).all()

    class TestScaledNoise(object):

        def test__scaled_background_noise__adds_to_input_noise(self):
            noise = np.array([1.0, 2.0, 3.0])
            background_noise = np.ones((3))

            hyper_image = hi.HyperImage(background_noise_scale=2.0)

            scaled_noise = hyper_image.hyper_noise_from_background_noise(noise=noise, background_noise=background_noise)

            assert (scaled_noise == np.array([3.0, 4.0, 5.0])).all()

        def test__variable_background_noise__adds_to_input_noise_different_for_each_value(self):
            noise = np.array([1.0, 2.0, 3.0])
            background_noise = np.array([5.0, 1.0, 3.0])

            hyper_image = hi.HyperImage(background_noise_scale=3.0)

            scaled_noise = hyper_image.hyper_noise_from_background_noise(noise=noise, background_noise=background_noise)

            assert (scaled_noise == np.array([16.0, 5.0, 12.0])).all()
