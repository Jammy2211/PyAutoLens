class MockPointSolver:
    def __init__(self, tracer, model_positions):
        self.lensing_obj = tracer
        self.model_positions = model_positions

    @property
    def tracer(self):
        return self.lensing_obj

    def solve(self, source_plane_coordinate):
        return self.model_positions
