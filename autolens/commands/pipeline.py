"""Reset Config."""

from autolens.commands.base import Base
from autolens import config


class Pipeline(Base):
    """Reset Config!"""

    def run(self):
        print("Available Pipelines:")
        from autolens.pipeline import pipeline

        pipelines = [value for value in pipeline.__dict__.values() if isinstance(value, pipeline.Pipeline)]
        print("\n".join([p.name for p in pipelines]))


if __name__ == "__main__":
    Pipeline(None).run()
