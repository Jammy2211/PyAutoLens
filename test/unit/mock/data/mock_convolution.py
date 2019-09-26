import autolens as al


class MockConvolver(al.Convolver):
    def __init__(self, mask, kernel, blurring_mask=None):
        super(MockConvolver, self).__init__(
            mask=mask, blurring_mask=blurring_mask, kernel=kernel
        )

    def convolver_with_blurring_mask_added(self, blurring_mask):
        return MockConvolver(mask=self.mask, kernel=self.kernel, blurring_mask=blurring_mask)
