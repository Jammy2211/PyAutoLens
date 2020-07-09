""""""

# %%
"""
Tutorial 2: Two Lens Galaxies
=============================

Up to now, all the images we've fitted had one lens galaxy. However, we saw in chapter 1 that our lens plane can
consist of multiple galaxies which each contribute to the strong lensing. Multi-galaxy systems are challenging to
model, because they add an extra 5-10 parameters to the non-linear search and, more problematically, the degeneracies
between the _MassProfile_'s of the two galaxies can be severe.

However, we can still break their analysis down using a pipeline and give ourselves a shot at getting a good lens
model. Here, we're going to fit a double lens system, fitting as much about each individual lens galaxy before fitting
them simultaneously.

Up to now, I've put a focus on pipelines being general. The pipeline we write in this example is going to be the
opposite, specific to the image we're modeling. Fitting multiple lens galaxies is really difficult and writing a
pipeline that we can generalize to many lenses isn't currently possible with __PyAutoLens__.
"""

# %%
""" AUTOFIT + CONFIG SETUP """

# %%
from autoconf import conf
from pyprojroot import here

workspace_path = str(here())
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
We'll use new strong lensing data, where:

 - There are two lens galaxy's whose _LightProfile_'s are both _EllipticalSersic_'s.
 - There are two lens galaxy's whose _MassProfile_'s are both _EllipticalIsothermal_'s.
 - The source galaxy's _LightProfile_ is an _EllipticalExponential_.
"""

# %%
from howtolens.simulators.chapter_3 import lens_x2_sersic_sie__source_exp

dataset_type = "chapter_3"
dataset_name = "lens_x2_sersic_sie__source_exp"
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

The *PhaseSettingsImaging* describe how the model is fitted to the data in the log likelihood function. We discussed
these in chapter 2, and a full description of all settings can be found in the example script:

 'autolens_workspace/examples/model/customize/settings.py'.

The settings chosen here are applied to all phases in the pipeline.
"""

# %%
settings = al.PhaseSettingsImaging(grid_class=al.Grid, sub_size=2)

# %%
"""
__Pipeline Approach__

Looking at the image, there are clearly two blobs of light corresponding to our two lens galaxies. The source's 
light is also pretty complex - the arcs don't posses the rotational symmetry we're used to seeing up to now. 
Multi-galaxy ray-tracing is just a lot more complicated, which means so is modeling it!

So, how can we break the lens modeling up? As follows:

 1) Fit and subtract the light of each lens galaxy individually.
 2) Use these results to initialize each lens galaxy's _MassProfile_.

So, with this in mind, we've written a pipeline composed of 4 phases:

 1) Fit the _LightProfile_ of the lens galaxy on the left of the image, at coordinates (0.0", -1.0").
 2) Fit the _LightProfile_ of the lens galaxy on the right of the image, at coordinates (0.0", 1.0").
 3) Use this lens-subtracted image to fit the source-galaxy's light. The _MassProfile_'s of the two lens 
 galaxies are fixed to (0.0", -1.0") and (0.0", 1.0").
 4) Fit all relevant parameters simultaneously, using priors from phases 1, 2 and 3.
"""

# %%
"""
__Pipeline_Setup_And_Tagging__:

For this pipeline the pipeline setup customizes:

 - If there is an external shear in the mass model or not.

The pipeline setup 'tags' the output path of a pipeline. For example, if 'no_shear' is True, the pipeline's output 
paths are 'tagged' with the string 'no_shear'.

This means you can run the same pipeline on the same data twice (with and without shear) and the results will go
to different output folders and thus not clash with one another!
"""

# %%
setup = al.PipelineSetup(no_shear=False, folders=["howtolens", "c3_t2_x2_galaxies"])

# %%
"""
__Pipeline Creation__

To create a _Pipeline_, we call a 'make_pipeline' function, which is written in its own Python script: 

 'tutorial_2_pipeline_x2_lens_galaxies.py'. 

Before we check it out, lets get the pipeline running, by importing the script, running the 'make_pipeline' function
to create the _Pipeline_ object and calling that objects 'run' function.

The 'folders' below specify the path the pipeline results are written to, which is:

 'autolens_workspace/output/howtolens/c3_t2_x2_galaxies/pipeline_name/setup_tag/phase_name/settings_tag'
"""

# %%
from howtolens.chapter_3_pipelines import tutorial_2_pipeline_x2_lens_galaxies

pipeline_x2_galaxies = tutorial_2_pipeline_x2_lens_galaxies.make_pipeline(
    setup=setup, settings=settings
)

# Uncomment to run.
# pipeline_x2_galaxies.run(dataset=imaging, mask=mask)

# %%
"""
Now, read through the 'tutorial_2_pipeline_x2_galaxies.py' pipeline, to get a complete view of how it works. 
Once you've done that, come back here and we'll wrap up this tutorial.
"""

# %%
"""
And, we're done. This pipeline takes a while to run, as is the nature of multi-galaxy modeling. Nevertheless, 
the techniques we've learnt above can be applied to systems with even more _Galaxy_'s albeit the increases in 
parameters will slow down the non-linear search. Here are some more Q&A's

 1) This system had two very similar lens galaxy's with comparable amounts of light and mass. How common is this? 
 Does it make it harder to model them?

Typically, a 2 galaxy system has 1 massive galaxy (that makes up some 80%-90% of the overall light and mass), 
accompanied by a smaller satellite. The satellite can't be ignored - it impacts the ray-tracing in a measureable way, 
but its a lot less degenerate with the 'main' lens galaxy. This means we can often model the  satellite with much 
simpler profiles (e.g. spherical profiles). So yes, multi-galaxy systems can often be easier to model.

 2) It got pretty confusing passing all those priors towards the end of the pipeline there, didn't it?

It does get confusing, I won't lie. This is why we made galaxies named objects - so that we could call them the 
'left_lens' and 'right_lens'. It still requires caution when writing the pipeline, but goes to show that if you name 
your galaxies sensibly you should be able to avoid errors, or spot them quickly when you make them.
"""
