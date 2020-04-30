from autogalaxy.pipeline.phase.dataset import analysis
from autolens.lens import ray_tracing


def last_result_with_use_as_hyper_dataset(results):

    if results.last is not None:
        for index, result in enumerate(reversed(results)):
            if result.use_as_hyper_dataset:
                return result


class Analysis(analysis.Analysis):
    def plane_for_instance(self, instance):
        raise NotImplementedError()

    def tracer_for_instance(self, instance):
        return ray_tracing.Tracer.from_galaxies(
            galaxies=instance.galaxies, cosmology=self.cosmology
        )
