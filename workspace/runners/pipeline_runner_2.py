from autofit import conf
from autolens.data import ccd
from autolens.data.plotters import ccd_plotters

import os

# Welcome to the pipeline runner. This tool allows you to load data on strong lenses, and pass it to pipelines for a
# strong lens analysis. To show you around, we'll load up some example data and run it through some of the example
# pipelines that come distributed with PyAutoLens.

# The runner is supplied as both this Python script and a Juypter notebook. Its up to you which you use - I personally
# prefer the python script as provided you keep it relatively small, its quick and easy to comment out different lens
# names and pipelines to set off different analyses. However, notebooks are a tidier way to manage visualization - so
# feel free to use notebooks. Or, use both for a bit, and decide your favourite!

# The pipeline runner is fairly self explanatory. Make sure to checkout the pipelines in the
#  _workspace/pipelines/examples/_ folder - they come with detailed descriptions of what they do! I hope that you'll
# expand on them for your own personal scientific needs

# Get the relative path to the config files and output folder in our workspace.
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

# It is convenient to specify the lens name as a string, so that if the pipeline is applied to multiple images we \
# don't have to change all of the path entries in the load_ccd_data_from_fits function below.

lens_name = 'no_lens_light_and_x2_source' # An example simulated image with lens light emission and a source galaxy.
pixel_scale = 0.1

ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/example/' + lens_name + '/image.fits',
                                       psf_path=path+'/data/example/'+lens_name+'/psf.fits',
                                       noise_map_path=path+'/data/example/'+lens_name+'/noise_map.fits',
                                       pixel_scale=pixel_scale)

ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

# Running a pipeline is easy, we simply import it from the pipelines folder and pass the lens data to its run function.
# Below, we'll' use a 3 phase example pipeline to fit the data with a parametric lens light, mass and source light
# profile. Checkout _workspace/pipelines/examples/lens_light_and_x1_source_parametric.py_' for a full description of
# the pipeline.

from workspace.pipelines.examples import no_lens_light_and_source_inversion

pipeline = no_lens_light_and_source_inversion.make_pipeline(pipeline_path='example/' + lens_name + '/')

pipeline.run(data=ccd_data)