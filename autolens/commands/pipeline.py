from autolens.commands.base import Base
from autolens import pipeline


class Pipeline(Base):

    def run(self):
        name = self.options['<name>']
        if name is not None:
            try:
                self.run_pipeline(pipeline.pipeline_dict[name])
            except KeyError:
                print("No pipeline called {} found".format(name))
        else:
            print("Available Pipelines:")
            print("\n".join(list(pipeline.pipeline_dict.keys())))

    def run_pipeline(self, pipeline):
        pass
