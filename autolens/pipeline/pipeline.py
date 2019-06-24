import autofit as af


class PipelineImaging(af.Pipeline):
    def run(self, data, mask=None, positions=None, data_name=None):
        def runner(phase, results):
            return phase.run(data=data, results=results, mask=mask, positions=positions)

        return self.run_function(
            runner,
            data_name
        )


class PipelinePositions(af.Pipeline):
    def run(self, positions, pixel_scale):
        def runner(phase, results):
            return phase.run(positions=positions, pixel_scale=pixel_scale,
                             results=results)

        return self.run_function(
            runner
        )
