import autofit as af
import autoarray as aa


class MockPointSolver:
    def __init__(self, model_positions):

        self.model_positions = model_positions

    def solve(self, lensing_obj, source_plane_coordinate, upper_plane_index=None):
        return self.model_positions
