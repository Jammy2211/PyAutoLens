""""""

# %%
"""
Tutorial 1: Lens and Source
===========================

As discussed in chapter 2, an effective strategy for modeling strong lenses is to break the model-fitting procedure
down into a series of phases, linking the results of the phases to guide the non-linear search as to where to
sample parameter space. In this chapter, we'll cover _Pipeline_'s, which provide a powerful means by which to
link together many phases.

In chapter 2, we fitted strong lens _Imaging_ which included the lens galaxy's light. We're going to fit this dataset
again (I promise, this is the last time!). However, now we're using pipelines, we can perform a different (and
significantly faster) model-fit.

The crucial point to note is that for many lenses the lens galaxy's light can be fitted and subtracted reasonable well
before we attempt to fit the source galaxy. This makes sense, as fitting the lens's light (which is an elliptical blob
of light in the centre of the imaging) looks nothing like the source's light (which is a ring of light)! Formally,
we would say that these two model components (the lens's light and source's light) are not covariant.

So, as a newly trained lens modeler, what does the lack of covariance between these parameters make you think?
Hopefully, you're thinking, why should I bother fitting the lens and source galaxy simultaneously? Surely we can
find the right regions of non-linear parameter space by fitting each separately first? This is what we're going to do
in this tutorial, using a pipeline composed of a modest 3 phases:

 1) Fit the lens galaxy's light, ignoring the source.
 2) Fit the source-galaxy's light (and therefore lens galaxy's mass), ignoring the len's light.
 3) Fit both simultaneously, using these results to initialize our starting location in parameter space.

Of course, given that we do not care for the errors in phases 1 and 2, we will set up our non-linear search to perform
sampling as fast as possible!
"""

# %%
""" AUTOFIT + CONFIG SETUP """

# %%
#%matplotlib inline
from autoconf import conf
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

conf.instance = conf.Config(
    config_path=f"{workspace_path}/howtolens/config",
    output_path=f"{workspace_path}/howtolens/output",
)

# %%
""" AUTOLENS + DATA SETUP """

# %%
#%matplotlib inline

import autolens as al
import autolens.plot as aplt

# %%
"""
We'll use strong lensing data, where:

 - The lens galaxy's _LightProfile_ is an _EllipticalSersic_.
 - The lens galaxy's _MassProfile_ is an _EllipticalIsothermal_.
 - The source galaxy's _LightProfile_ is an _EllipticalExponential_.
"""

# %%
from autolens_workspace.howtolens.simulators.chapter_3 import (
    light_sersic__mass_sie__source_exp,
)

dataset_type = "chapter_3"
dataset_name = "light_sersic__mass_sie__source_exp"
dataset_path = f"{workspace_path}/howtolens/dataset/{dataset_type}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    psf_path=f"{dataset_path}/psf.fits",
    pixel_scales=0.1,
)

# %%
"""
We need to choose our mask for the analysis. Given the lens light is present in the image we'll need to include all 
of its light in the central regions of the image, so lets use a circular mask.
"""

# %%
mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
__Settings__

The _SettingsPhaseImaging_ describe how the model is fitted to the data in the log likelihood function. We discussed
these in chapter 2, and a full description of all settings can be found in the example script:

 'autolens_workspace/examples/model/customize/settings.py'.

The settings chosen here are applied to all phases in the pipeline.
"""

# %%
settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

# %%
"""
__Pipeline_Setup__:

Pipelines can contain _Setup_ objects, which customize how different aspects of the model are fitted. 

First, we create a _SetupLightSersic_ which customizes:

 - If the centre of the lens light profile is manually input and fixed for modeling.

In this example we do not fix the centre of the _EllipticalSersic_ light model to a specific value.
"""

# %%
setup_light = al.SetupLightSersic(light_centre=None)

# %%
"""
This pipeline also uses a _SetupMass_, which customizes:

 - If there is an _ExternalShear_ in the mass model or not.
"""

# %%
setup_mass = al.SetupMassTotal(no_shear=False)

# %%
"""
Next, we create a _SetupSourceSersic_ which does not customize the pipeline behaviour except for tagging (see below).
"""

# %%
setup_source = al.SetupSourceSersic()

"""
_Pipeline Tagging_

The _Setup_ objects are input into a _SetupPipeline_ object, which is passed into the pipeline and used to customize
the analysis depending on the setup. This includes tagging the output path of a pipeline. For example, if 'no_shear' 
is True, the pipeline's output paths are 'tagged' with the string 'no_shear'.

This means you can run the same pipeline on the same data twice (with and without shear) and the results will go
to different output folders and thus not clash with one another!

The 'folders' below specify the path the pipeline results are written 

 'autolens_workspace/howtolens/output/c3_t1_lens_and_source/pipeline__light_and_source'

The redshift of the lens and source galaxies are also input (see 'examples/model/customimze/redshift.py') for a 
description of what inputting redshifts into **PyAutoLens** does.
"""

# %%
setup = al.SetupPipeline(
    folders=["c3_t1_lens_and_source"],
    redshift_lens=0.5,
    redshift_source=1.0,
    setup_mass=setup_mass,
    setup_source=setup_source,
)

# %%
"""
__Pipeline Creation__

To create a _Pipeline_, we call a 'make_pipeline' function, which is written in its own Python script: 

 'tutorial_1_pipeline_lens_and_source.py'. 
    
Before we check it out, lets get the pipeline running, by importing the script, running the 'make_pipeline' function
to create the _Pipeline_ object and calling that objects 'run' function.

The 'folders' below specify the path the pipeline results are written to, which is:

 'autolens_workspace/output/howtolens/c3_t1_lens_and_source/pipeline_name/setup_tag/phase_name/settings_tag'
"""

# %%
from autolens_workspace.howtolens.chapter_3_pipelines import (
    tutorial_1_pipeline_lens_and_source,
)

pipeline_lens_and_source = tutorial_1_pipeline_lens_and_source.make_pipeline(
    setup=setup, settings=settings
)

# Uncomment to run.
# pipeline_lens_and_source.run(dataset=imaging, mask=mask)

# %%
"""
Okay, good job, we're running our first pipeline in __PyAutoLens__! But what does it *actually* do? Well, to find that out, 
go to the script 'tutorial_1_pipeline_lens_and_source.py', which contains a full description of the pipeline, as well 
as an overview of the tools we use to write the most general pipelines possible. Once you're done, come back to this 
pipeline runner script and we'll wrap up tutorial 1.
"""

# %%
"""
And there we have it, a pipeline that breaks the analysis of the lens and source galaxy into 3 simple phases. This 
approach is much faster than fitting the lens and source simultaneously from the beginning. Instead of asking you 
questions at the end of this chapter's tutorials, I'm gonna give a Q&A - this'll hopefully get you thinking about how 
to approach pipeline writing.

 1) Can this pipeline really be generalized to any lens? Surely the radii of the mask depends on the lens and source 
 galaxies?

Whilst this is true, we've chosen a mask radii above that is 'excessive' and masks out a lot more of the image than 
just the source (which, in terms of run-time, is desirable). Thus, provided you know the Einstein radius distribution 
of your lens sample, you can choose mask radii that will masks out every source in your sample adequately (and even if 
some of the source is still there, who cares? The fit to the lens galaxy will be okay).
"""
