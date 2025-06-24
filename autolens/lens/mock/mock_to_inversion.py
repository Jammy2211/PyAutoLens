from typing import Dict, Optional


class MockTracerToInversion:
    def __init__(
        self,
        tracer,
        image_plane_mesh_grid_pg_list=None,
    ):
        self.tracer = tracer

        self.image_plane_mesh_grid_pg_list = image_plane_mesh_grid_pg_list

    def image_plane_mesh_grid_pg_list(self):
        return self.image_plane_mesh_grid_pg_list
