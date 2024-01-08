import autoarray as aa

from autolens.lens.to_inversion import TracerToInversion

from autolens.lens.mock.mock_to_inversion import MockTracerToInversion


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

    @property
    def tracer_to_inversion(self) -> MockTracerToInversion:

        return MockTracerToInversion(
            tracer=self.tracer,
            image_plane_mesh_grid_pg_list=self.tracer.image_plane_mesh_grid_pg_list,
        )
