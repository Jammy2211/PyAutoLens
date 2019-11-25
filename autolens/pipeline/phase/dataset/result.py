from autolens.pipeline.phase import abstract


class Result(abstract.result.Result):
    @property
    def most_likely_fit(self):

        hyper_image_sky = self.analysis.hyper_image_sky_for_instance(
            instance=self.instance
        )

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=self.instance
        )

        return self.analysis.masked_imaging_fit_for_tracer(
            tracer=self.most_likely_tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    @property
    def mask(self):
        return self.most_likely_fit.mask

    @property
    def positions(self):
        return self.most_likely_fit.masked_dataset.positions

    @property
    def pixelization(self):
        for galaxy in self.most_likely_fit.tracer.galaxies:
            if galaxy.pixelization is not None:
                return galaxy.pixelization

    @property
    def most_likely_pixelization_grids_of_planes(self):
        return self.most_likely_tracer.sparse_image_plane_grids_of_planes_from_grid(
            grid=self.most_likely_fit.grid
        )
