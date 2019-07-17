import autofit as af
from autolens import exc

class PipelineImaging(af.Pipeline):

    def __init__(self, pipeline_name, *phases, hyper_mode=False):

        super(PipelineImaging, self).__init__(pipeline_name, *phases)

        self.hyper_mode = hyper_mode

    def run(self, data, mask=None, positions=None, data_name=None):

        if self.hyper_mode and mask is None:
            raise exc.PhaseException(
                'The pipeline is running in hyper mode, but has not received an input mask. Add'
                'a mask to the run function of the pipeline (e.g. pipeline.run(data=data, mask=mask)')

        def runner(phase, results):
            return phase.run(data=data, results=results, mask=mask, positions=positions)

        return self.run_function(runner, data_name)


class PipelinePositions(af.Pipeline):
    def run(self, positions, pixel_scale):
        def runner(phase, results):
            return phase.run(
                positions=positions, pixel_scale=pixel_scale, results=results
            )

        return self.run_function(runner)
