from autolens.pipeline import profile_pipeline
from autolens.pipeline import source_only_pipeline


class PipelineTuple(object):
    def __init__(self, module):
        self.make = module.make
        self.doc = module.__doc__

    @property
    def short_doc(self):
        return self.doc.split('\n')[1]


pipeline_dict = {}


def add(module):
    """
    Parameters
    ----------
    module: {name, make}
    """
    pipeline_dict[module.name] = PipelineTuple(module)


add(profile_pipeline)
add(source_only_pipeline)
