import logging

logger = logging.getLogger(__name__)


class Pipeline(object):
    def __init__(self, name, *phases):
        """

        Parameters
        ----------
        name: str
            The name of this pipeline
        phases: [ph.Phase]
            Phases
        """
        self.name = name
        self.phases = phases

    def run(self, image):
        from autolens.pipeline import phase as ph
        results = []
        for i, phase in enumerate(self.phases):
            logger.info("Running Phase {} (Number {})".format(phase.name, i))
            results.append(phase.run(image, ph.ResultsCollection(results)))
        return results

    def __add__(self, other):
        """
        Compose two pipelines

        Parameters
        ----------
        other: Pipeline
            Another pipeline

        Returns
        -------
        composed_pipeline: Pipeline
            A pipeline that runs all the  phases from this pipeline and then all the phases from the other pipeline
        """
        return Pipeline("{} + {}".format(self.name, other.name), *(self.phases + other.phases))
