import numpy as np

from auto_lens.imaging import simulate

class TestConstructor(object):

    def test__setup_with_all_features_off(self):

        image = np.array(([0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0]))

        psf = np.array(([0.0, 1.0, 0.0],
                        [1.0, 2.0, 1.0],
                        [0.0, 1.0, 0.0]))

        mock_image = simulate.SimulateImage(data=image, pixel_scale=0.1, psf=psf)

        assert (mock_image == np.array(([1.0, 1.0, 1.0],
                                        [2.0, 2.0, 2.0],
                                        [3.0, 3.0, 3.0]))).all()

        assert (mock_image.psf == np.array(([0.0, 1.0, 0.0],
                                            [1.0, 2.0, 1.0],
                                            [0.0, 1.0, 0.0]))).all()

        assert (mock_image.pixel_scale == 0.1)

