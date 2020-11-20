from autogalaxy.pipeline.phase.dataset import result as ag_result
from autolens.pipeline.phase.abstract import result


class Result(result.Result, ag_result.Result):
    @property
    def mask(self):
        return self.max_log_likelihood_fit.mask

    @property
    def positions(self):
        return self.max_log_likelihood_fit.masked_dataset.positions

    @property
    def pixelization(self):
        for galaxy in self.max_log_likelihood_fit.tracer.galaxies:
            if galaxy.pixelization is not None:
                return galaxy.pixelization

    @property
    def max_log_likelihood_pixelization_grids_of_planes(self):
        return (
            self.max_log_likelihood_tracer.sparse_image_plane_grids_of_planes_from_grid(
                grid=self.max_log_likelihood_fit.grid
            )
        )
