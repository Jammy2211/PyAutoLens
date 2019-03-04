from autofit import conf
from autolens.data import ccd
from autolens.data.plotters import ccd_plotters

import os
import sys

### NOTE - if you have not already, complete the setup in 'workspace/runners/cosma/setup' before continuing with this
### cosma pipeline script.

# Welcome to the Cosma pipeline runner. Hopefully, you're familiar with runners at this point, and have been using them
# with PyAutoLens to model lenses on your laptop. If not, I'd recommend you get used to doing that, before trying to
# run PyAutoLens on a super-computer. You need some familiarity with the software and lens modeling before trying to
# model a large sample of lenses on a supercomputer!

# If you are ready, then let me take you through the Cosma runner. It is remarkably similar to the ordinary pipeline
# runners you're used to, however it makes a few changes for running jobs on cosma:

# 1) The data path is over-written to the path '/cosma5/data/durham/cosma_username/autolens/data' as opposed to the
#    workspace. As we saw in the setup, on cosma we don't store our data in our workspace.

# 2) The output path is over-written to the path '/cosma5/data/durham/cosma_username/autolens/output' as opposed to
#    the workspace. This is for the same reason as the data.

# Given your username is where your data is stored, you'll need to put your cosma username here.
cosma_username = 'pdtw24'
cosma_data_path = '/cosma5/data/autolens/'

# Get the relative path to the config files and output folder in our workspace.
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path, and override the output path with the Cosma path.
conf.instance = conf.Config(config_path=path+'config', output_path=cosma_data_path+'output_'+cosma_username)

# Lets take a look at a Cosma batch script, which can be found at 'workspace/runners/cosma/batch/pipeline_runner_cosma'.
# When we submit a PyAutoLens job to Cosma, we submit a 'batch' of jobs, whereby each job will run on one CPU of Cosma.
# Thus, if our lens sample contains, lets say, 4 lenses, we'd submit 4 jobs at the same time where each job applies
# our pipeline to each image.

# The fifth line of this batch script - '#SBATCH --array=1-4' is what species this. Its telling Cosma we're going to
# run 4 jobs, and the id's of those jobs will be numbered from 1 to 4. Infact, these ids are passed to this runner,
# and we'll use them to ensure that each jobs loads a different image. Lets get the cosma array id for our job.
cosma_array_id = int(sys.argv[1])

# Now, I just want to really drive home what the above line is doing. For every job we run on Cosma, the cosma_array_id
# will be different. That is, job 1 will get a cosma_array_id of 1, job 2 will get an id of 2, and so on. This is our
# only unique identifier of every job, thus its our only hope of specifying for each job which image they load!

# Fortunately, we're used to specifying the lens name as a string, so that our pipeline can be applied to multiple
# images with ease. On Cosma, we can apply the same logic, but put these strings in a list such that each Cosma job
# loads a different lens name based on its ID. neat, huh?

lens_name = []
lens_name.append('') # Task number beings at 1, so keep index 0 blank
lens_name.append('example_lens_1') # Index 1
lens_name.append('example_lens_2') # Index 2
lens_name.append('example_lens_3') # Index 3
lens_name.append('example_lens_4') # Index 4

pixel_scale = 0.2 # Make sure your pixel scale is correct!

# We now use the lens_name list to load the image on each job, noting that in this example I'm assuming our lenses are
# on the Cosma data directory folder called 'example_cosma'
data_folder = 'example_cosma'
data_path = cosma_data_path + 'data/' + data_folder + '/'

ccd_data = ccd.load_ccd_data_from_fits(
    image_path=data_path + lens_name[cosma_array_id] + '/image.fits',
    psf_path=data_path + lens_name[cosma_array_id]+'/psf.fits',
    noise_map_path=data_path + lens_name[cosma_array_id]+'/noise_map.fits',
    pixel_scale=pixel_scale)

# Running a pipeline is exactly the same as we're used to. We import it, make it, and run it, noting that we can
# use the lens_name's to ensure each job outputs its results to a different directory.

from workspace.pipelines.examples import no_lens_light_and_x2_source_parametric

pipeline = no_lens_light_and_x2_source_parametric.make_pipeline(pipeline_path=lens_name[cosma_array_id] + '/')

pipeline.run(data=ccd_data)

# Finally, its worth us going through a batch script in detail, line by line, as you may we need to change different
# parts of this script to use different runners. Therefore, checkout the 'doc' file in the batch folder.