import autoarray as aa

class MockConvolver(aa.Convolver):
    def __init__(self, mask, kernel):
        super(MockConvolver, self).__init__(
            mask=mask, kernel=kernel.in_2d
        )