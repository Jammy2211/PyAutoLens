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
        Compose two runners

        Parameters
        ----------
        other: Pipeline
            Another pipeline

        Returns
        -------
        composed_pipeline: Pipeline
            A pipeline that runs all the  phases from this pipeline and then all the phases from the other pipeline
        """
        return self.__class__("{} + {}".format(self.pipeline_name, other.pipeline_name), *(self.phases + other.phases))


class PipelineImaging(Pipeline):

    def __init__(self, pipeline_name, *phases):
        super(PipelineImaging, self).__init__(pipeline_name, *phases)

    def run(self, data, mask=None, positions=None, skip_optimizer=False):

        from autolens.pipeline import phase as ph
        results = []
        for i, phase in enumerate(self.phases):
            logger.info("Running Phase {} (Number {})".format(phase.phase_name, i))
            results.append(phase.run(data=data, previous_results=ph.ResultsCollection(results), mask=mask,
                                     positions=positions, skip_optimizer=skip_optimizer))
        return results


class PipelinePositions(Pipeline):

    def __init__(self, pipeline_name, *phases):
        super(PipelinePositions, self).__init__(pipeline_name, *phases)

    def run(self, positions, pixel_scale, skip_optimizer=False):
        from autolens.pipeline import phase as ph

        results = []
        for i, phase in enumerate(self.phases):
            logger.info("Running Phase {} (Number {})".format(phase.phase_name, i))
            results.append(phase.run(positions=positions, pixel_scale=pixel_scale,
                                     previous_results=ph.ResultsCollection(results), skip_optimizer=skip_optimizer))
        return results
