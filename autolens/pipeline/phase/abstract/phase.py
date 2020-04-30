import autogalaxy as ag

from autolens.pipeline.phase.abstract.result import Result


# noinspection PyAbstractClass


class AbstractPhase(ag.AbstractPhase):

    Result = Result

    def make_result(self, result, analysis):

        return self.Result(
            samples=result.samples,
            previous_model=result.previous_model,
            analysis=analysis,
            optimizer=self.optimizer,
            use_as_hyper_dataset=self.use_as_hyper_dataset,
        )
