from autolens.pipeline import profile_pipeline
from autolens.pipeline import source_only_pipeline
from collections import namedtuple

PipelineTuple = namedtuple("PipelineTuple", ["make", "doc"])

pipeline_dict = {}


def add(module):
    """
    Parameters
    ----------
    module: {name, make}
    """
    pipeline_dict[module.name] = PipelineTuple(module.make, module.__doc__)


add(profile_pipeline)
add(source_only_pipeline)
