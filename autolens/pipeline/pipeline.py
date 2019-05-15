from autofit.tools.pipeline import Pipeline


class PipelineImaging(Pipeline):
    def run(self, data, mask=None, positions=None, data_name=None, assert_optimizer_pickle_matches=False):
        def runner(phase, results):
            return phase.run(data=data, results=results, mask=mask, positions=positions)

        return self.run_function(runner, data_name, assert_optimizer_pickle_matches=assert_optimizer_pickle_matches)


class PipelinePositions(Pipeline):
    def run(self, positions, pixel_scale, assert_optimizer_pickle_matches=True):
        def runner(phase, results):
            return phase.run(positions=positions, pixel_scale=pixel_scale, results=results)

        return self.run_function(runner, assert_optimizer_pickle_matches=assert_optimizer_pickle_matches)
