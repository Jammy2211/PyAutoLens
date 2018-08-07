from autolens.commands.base import Base


class Pipeline(Base):

    def run(self):
        from autolens.pipeline import pipeline
        pipelines = {value.name: value for value in pipeline.__dict__.values() if isinstance(value, pipeline.Pipeline)}
        name = self.options['<name>']
        if name is not None:
            try:
                self.run_pipeline(pipelines[name])
            except KeyError:
                print("No pipeline called {} found".format(name))
        else:
            print("Available Pipelines:")
            print("\n".join(list(pipelines.keys())))

    def run_pipeline(self, pipeline):
        pass
