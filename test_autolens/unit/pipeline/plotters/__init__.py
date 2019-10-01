class PipelineTuple(object):
    def __init__(self, module):
        self.make = module.make
        self.__doc__ = module.__doc__

    @property
    def short_doc(self):
        return self.__doc__.split("\n")[1]

    @property
    def doc(self):
        return self.__doc__.replace("  ", "").replace("\n", " ")


pipeline_dict = {}


def add(module):
    """
    Parameters
    ----------
    module: {analysis_path, make}
    """
    pipeline_dict[module.pipeline_name] = PipelineTuple(module)


#  Add pipeline modules here.
# add(initialize)


class TestPipeline(object):
    # noinspection PyMethodMayBeStatic
    def run(self, image):
        print(image)
