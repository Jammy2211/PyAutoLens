# %%
"""
Tutorial 8: Results
===================

Once a phase has completed running, it results a `Result` object, which in the previous tutorials we used to plot
the maximum log likelihood fit of the modoel-fits. Lets take a more detailed look at what else the results contains.
"""

# %%
#%matplotlib inline

from pyprojroot import here

workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt
import autofit as af

# %%
"""
Lets reperform the model-fit from tutorial 1 to get a results object, provided you didn`t delete the results on
your hard-disk this should simply reload them into this Pythons script.
"""

# %%
dataset_name = "mass_sis__source_exp"
dataset_path = path.join("dataset", "howtolens", "chapter_2", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

phase = al.PhaseImaging(
    search=af.DynestyStatic(
        path_prefix="howtolens", name="phase_t1_non_linear_search", n_live_points=40
    ),
    settings=al.SettingsPhaseImaging(
        settings_masked_imaging=al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2)
    ),
    galaxies=dict(
        lens_galaxy=al.GalaxyModel(redshift=0.5, mass=al.mp.SphericalIsothermal),
        source_galaxy=al.GalaxyModel(redshift=1.0, bulge=al.lp.SphericalExponential),
    ),
)

result = phase.run(dataset=imaging, mask=mask)

# %%
"""
In the previous tutorials, we saw that this result contains the maximum log likelihood tracer and fit, which provide
a fast way to visualize the result.

(Uncomment the line below to pllot the tracer).
"""
# aplt.Tracer.subplot_tracer(
#    tracer=result.max_log_likelihood_tracer, grid=mask.geometry.unmasked_grid
# )
aplt.FitImaging.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

# %%
"""
The result contains a lot more information about the model-fit. For example, its `Samples` object contains the complete
set of `NonLinearSearch` samples, for example every set of parameters evaluated, their log likelihoods and so on,
which are used for computing information about the model-fit such as the error on every parameter.
"""
print(result.samples)
print(result.samples.parameters)
print(result.samples.log_likelihoods)

# %%
"""
However, we are not going into any more detail on the result variable in this tutorial, or in the **HowToLens** lectures.

A comprehensive description of the results can be found at the following script:

 `autolens_workspace/examples/model/result.py`

"""

# %%
"""
__Aggregator__

Once a phase has completed running, we have a set of results on our hard disk we manually inspect and analyse. 
Alternatively, we can return the results from the phase.run() method and manipulate them in a Python script.  

However, imagine your dataset is large and consists of many images of strong lenses. You analyse each image 
individually using the same phase, producing a large set of results on your hard disk corresponding to the full sample.
That will be a lot of paths and directories to navigate! At some point, there`ll be too many results for it to be
a sensible use of your time to analyse the results by sifting through the outputs on your hard disk.

PyAutoFit`s aggregator tool allows us to load results in a Python script or, more impotantly, a Jupyter notebook.
All we have to do is point the aggregator to the output directory from which we want to load results, which in this c
ase will be the results of the first `NonLinearSearch` of this chapter.
"""

# %%
"""
To set up the aggregator we simply pass it the folder of the results we want to load.
"""

# %%
output_path = f"output"
agg = af.Aggregator(directory=str(output_path))
agg = agg.filter(agg.phase == "phase_t1_non_linear_search")

# %%
"""
We get the output of the results of the model-fit performed in tutorial 1, given that is the directory we point too. 
This gives us a list with 1 entry, the list would have more entries if there were more results in the path.
"""

# %%
samples = list(agg.values("samples"))

# %%
"""
From here, we can inspect results as we please, for example printing the maximum log likelihood model of the phase.
"""

# %%
print(samples[0].max_log_likelihood_vector)

# %%
"""
Again, we won't go into any more detail on the aggregator in this tutorial. For those of you modeling large samples of
lenses for who the tool will prove useful, checkout the full set of aggregator tutorials which can be found at the 
location `autolens_workspace/advanced`aggregator`. Here, you`ll learn how to:

 - Use the aggregator to filter out results given a phase name or input string.
 - Use the Samples to produce many different results from the fit, including error estimates on parameters and 
      plots of the probability density function of parameters in 1D and 2D.
 - Reproduce visualizations of results, such as a tracer`s images or the fit to a lens dataset.

Even if you are only modeling a small sample of lenses, if you anticipate using **PyAutoLens** for the long-term I 
strongly recommend you begin using the aggregator to inspect and analyse your result. This is because it makes it 
simple to perform all analyse in a Jupyter notebook, which as you already know is a flexible and versatile way to check 
results and make figures.

In HowToLelens, the main purpose of this tutorial was to make sure that you are aware of the aggregator`s existance, 
and now you are!
"""
