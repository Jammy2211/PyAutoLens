from typing import Dict, Optional


class MockTracerToInversion:
    def __init__(
        self,
        tracer,
        sparse_image_plane_grid_pg_list=None,
        profiling_dict: Optional[Dict] = None,
    ):

        self.tracer = tracer

        self.sparse_image_plane_grid_pg_list = sparse_image_plane_grid_pg_list

        self.profiling_dict = profiling_dict

    def sparse_image_plane_grid_pg_list(self):

        return self.sparse_image_plane_grid_pg_list
