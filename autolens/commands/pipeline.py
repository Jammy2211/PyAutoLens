from autolens.commands.base import Base


class Pipeline(Base):

    def run(self):
        print("Available Pipelines:")
        from autolens.pipeline import pipeline

        pipelines = [value for value in pipeline.__dict__.values() if isinstance(value, pipeline.Pipeline)]
        print("\n".join([p.name for p in pipelines]))


if __name__ == "__main__":
    Pipeline(None).run()
