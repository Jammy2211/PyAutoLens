import numpy as np

from autolens.model.hyper import hyper_data as hi

class TestHyperImageSky(object):

    def test__scale_sky_in_image__increases_all_image_values(self):

        image = np.array([1.0, 2.0, 3.0])

        hyper_sky = hi.HyperImageSky(background_sky_scale=10.0)

        scaled_image = hyper_sky.image_scaled_sky_from_image(image=image)

        assert (scaled_image == np.array([11.0, 12.0, 13.0])).all()

class TestHyperNoiseMapBackground(object):

    def test__scaled_background_noise__adds_to_input_noise(self):

        noise_map = np.array([1.0, 2.0, 3.0])

        hyper_background_noise_map = hi.HyperNoiseBackground(background_noise_scale=2.0)

        hyper_noise_map = hyper_background_noise_map.noise_map_scaled_noise_from_noise_map(
            noise_map=noise_map)

        assert (hyper_noise_map == np.array([3.0, 4.0, 5.0])).all()

        hyper_noise_map_background = hi.HyperNoiseBackground(background_noise_scale=3.0)

        scaled_noise = hyper_noise_map_background.noise_map_scaled_noise_from_noise_map(
            noise_map=noise_map)

        assert (scaled_noise == np.array([4.0, 5.0, 6.0])).all()
