from autofit import conf
from autofit.tools.pipeline import Pipeline


class PipelineImaging(Pipeline):
    def run(self, data, mask=None, positions=None):
        def runner(phase, results):
            with open("{}/{}{}/.metadata".format(conf.instance.output_path, phase.phase_path, phase.phase_name),
                      "w+") as f:
                f.write("pipeline={}\nphase={}\nlens={}".format(self.pipeline_name, phase.phase_name, data.name))
            return phase.run(data=data, results=results, mask=mask, positions=positions)

        return self.run_function(runner)


class PipelinePositions(Pipeline):
    def run(self, positions, pixel_scale):
        def runner(phase, results):
            return phase.run(positions=positions, pixel_scale=pixel_scale, results=results)

        return self.run_function(runner)
