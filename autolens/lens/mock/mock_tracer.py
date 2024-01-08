class MockTracer:
    def __init__(
        self, traced_grid_2d_list_from=None, image_plane_mesh_grid_pg_list=None
    ):
        self.image_plane_mesh_grid_pg_list = image_plane_mesh_grid_pg_list
        self._traced_grid_2d_list_from = traced_grid_2d_list_from

    def traced_grid_2d_list_from(self, grid):
        return self._traced_grid_2d_list_from


class MockTracerPoint(MockTracer):
    def __init__(
        self,
        image_plane_mesh_grid_pg_list=None,
        traced_grid=None,
        attribute=None,
        profile=None,
        magnification=None,
        einstein_radius=None,
        einstein_mass=None,
    ):
        super().__init__(image_plane_mesh_grid_pg_list=image_plane_mesh_grid_pg_list)

        self.positions = traced_grid

        self.attribute = attribute
        self.profile = profile

        self.magnification = magnification
        self.einstein_radius = einstein_radius
        self.einstein_mass = einstein_mass

    @property
    def planes(self):
        return [0, 1]

    def deflections_yx_2d_from(self):
        pass

    def extract_attribute(self, cls, attr_name):
        return [self.attribute]

    def extract_profile(self, profile_name):
        return self.profile

    def traced_grid_2d_list_from(self, grid, plane_index_limit=None):
        return [self.positions]

    def magnification_2d_via_hessian_from(self, grid, deflections_func=None):
        return self.magnification

    def einstein_radius_from(self, grid):
        return self.einstein_radius

    def einstein_mass_angular_from(self, grid):
        return self.einstein_mass
