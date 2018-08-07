from autolens.pipeline import profile_pipeline
from autolens.pipeline import source_only_pipeline

pipeline_dict = {}


def add(module):
    pipeline_dict[module.name] = module.make


add(profile_pipeline)
add(source_only_pipeline)
