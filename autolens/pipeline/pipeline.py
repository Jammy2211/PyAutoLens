from autofit.tools.pipeline import Pipeline


class PipelineImaging(Pipeline):
    def run(self, data, mask=None, positions=None):
        def runner(phase, results):
            return phase.run(data=data, previous_results=results, mask=mask, positions=positions)

        return self.run_function(runner)


class PipelinePositions(Pipeline):
    def run(self, positions, pixel_scale):
        def runner(phase, results):
            return phase.run(positions=positions, pixel_scale=pixel_scale, previous_results=results)

        return self.run_function(runner)
