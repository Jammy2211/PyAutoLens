import autolens as al


class MockConvolver(al.Convolver):
    def __init__(self, mask, psf, blurring_mask=None):
        super(MockConvolver, self).__init__(
            mask=mask, blurring_mask=blurring_mask, psf=psf
        )

    def convolver_with_blurring_mask_added(self, blurring_mask):
        return MockConvolver(mask=self.mask, psf=self.psf, blurring_mask=blurring_mask)
