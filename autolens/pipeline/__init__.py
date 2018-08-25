from pipelines import profile_pipeline, source_only_pipeline


class PipelineTuple(object):
    def __init__(self, module):
        self.make = module.make
        self.__doc__ = module.__doc__

    @property
    def short_doc(self):
        return self.__doc__.split('\n')[1]

    @property
    def doc(self):
        return self.__doc__.replace("  ", "").replace("\n", " ")


pipeline_dict = {}


def add(module):
    """
    Parameters
    ----------
    module: {phase_name, make}
    """
    pipeline_dict[module.name] = PipelineTuple(module)


add(profile_pipeline)
add(source_only_pipeline)
