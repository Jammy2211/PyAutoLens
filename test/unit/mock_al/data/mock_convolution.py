import autolens as al


class MockConvolver(al.Convolver):
    def __init__(self, mask, kernel, blurring_mask=None):
        super(MockConvolver, self).__init__(
            mask=mask, kernel=kernel.in_2d
        )

    def convolver_with_blurring_mask_added(self, blurring_mask):
        return MockConvolver(
            mask=self.mask, kernel=self.kernel.in_2d, blurring_mask=blurring_mask
        )
