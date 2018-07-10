from src.imaging import mask as msk
from src.imaging import image as im
import numpy as np


class TestMaskedImage(object):
    def test_attributes(self):
        psf = im.PSF(np.ones((1, 1)), 1)
        image = im.Image(np.ones((3, 3)), effective_exposure_time=1., pixel_scale=1., psf=psf,
                         background_noise=np.ones((3, 3)), poisson_noise=np.ones((3, 3)))

        mask = msk.Mask(np.array([[True, False, True],
                                  [False, False, False],
                                  [True, False, True]]))

        masked_image = mask.mask_image(image)

        assert image.effective_exposure_time == masked_image.effective_exposure_time
        assert image.pixel_scale == masked_image.pixel_scale
        assert image.psf == masked_image.psf
        assert (image.background_noise == masked_image.background_noise).all()
        assert (image.poisson_noise == masked_image.poisson_noise).all()
