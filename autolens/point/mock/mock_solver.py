from typing import Optional


class MockPointSolver:
    def __init__(self, model_positions):
        self.model_positions = model_positions

    def solve(
        self,
        tracer,
        source_plane_coordinate,
        plane_redshift: Optional[float] = None,
    ):
        return self.model_positions
