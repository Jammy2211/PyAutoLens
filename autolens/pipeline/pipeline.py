from autolens import conf
import logging

logger = logging.getLogger(__name__)


class Pipeline(object):

    def __init__(self, pipeline_name, *phases):
        """

        Parameters
        ----------
        pipeline_name: str
            The phase_name of this pipeline
        """
        self.pipeline_name = pipeline_name
        self.phases = phases

    def __add__(self, other):
        """
        Compose two pipelines

        Parameters
        ----------
        other: PipelineImaging
            Another pipeline

        Returns
        -------
        composed_pipeline: PipelineImaging
            A pipeline that runs all the  phases from this pipeline and then all the phases from the other pipeline
        """
        return self.__class__("{} + {}".format(self.pipeline_name, other.pipeline_name), *(self.phases + other.phases))


class PipelineImaging(Pipeline):

    def __init__(self, pipeline_name, *phases):
        super(PipelineImaging, self).__init__(pipeline_name, *phases)

    def run(self, image):

        from autolens.pipeline import phase as ph
        results = []
        for i, phase in enumerate(self.phases):
            logger.info("Running Phase {} (Number {})".format(phase.phase_name, i))
            if not isinstance(phase, ph.HyperOnly):
                results.append(phase.run(image, ph.ResultsCollection(results)))
            elif isinstance(phase, ph.HyperOnly):
                results[-1].hyper = phase.hyper_run(image, ph.ResultsCollection(results))
        return results


class PipelinePositions(Pipeline):

    def __init__(self, pipeline_name, *phases):
        super(PipelinePositions, self).__init__(pipeline_name, *phases)

    def run(self, positions, pixel_scale):
        from autolens.pipeline import phase as ph

        results = []
        for i, phase in enumerate(self.phases):
            logger.info("Running Phase {} (Number {})".format(phase.phase_name, i))
            results.append(phase.run(positions, pixel_scale, ph.ResultsCollection(results)))
        return results


def setup_pipeline_path(pipeline_name):
    conf.instance.output_path = "{}/".format(conf.instance.output_path) + pipeline_name + '/'
