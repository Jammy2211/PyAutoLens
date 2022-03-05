import autoarray as aa


class MockFitImaging(aa.m.MockFitImaging):
    def __init__(
        self,
        tracer=None,
        dataset=aa.m.MockDataset(),
        inversion=None,
        noise_map=None,
        grid=None,
        blurred_image=None,
    ):

        super().__init__(
            dataset=dataset,
            inversion=inversion,
            noise_map=noise_map,
            blurred_image=blurred_image,
        )

        self.tracer = tracer
        self.grid = grid
