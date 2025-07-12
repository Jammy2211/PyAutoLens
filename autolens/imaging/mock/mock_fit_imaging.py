import autoarray as aa

from autolens.lens.mock.mock_to_inversion import MockTracerToInversion


class MockFitImaging(aa.m.MockFitImaging):
    def __init__(
        self,
        tracer=None,
        dataset=None,
        inversion=None,
        noise_map=None,
        grid=None,
        blurred_image=None,
    ):

        dataset = dataset or aa.m.MockDataset()

        super().__init__(
            dataset=dataset,
            inversion=inversion,
            noise_map=noise_map,
            blurred_image=blurred_image,
        )

        self._grid = grid
        self.tracer = tracer

    @property
    def grid(self):

        if self._grid is not None:
            return self._grid

        return super().grids.lp

    @property
    def grids(self) -> aa.GridsInterface:

        return aa.GridsInterface(
            lp=self.grid,
            pixelization=self.grid,
        )

    @property
    def tracer_to_inversion(self) -> MockTracerToInversion:

        return MockTracerToInversion(
            tracer=self.tracer,
            image_plane_mesh_grid_pg_list=self.tracer.image_plane_mesh_grid_pg_list,
        )
