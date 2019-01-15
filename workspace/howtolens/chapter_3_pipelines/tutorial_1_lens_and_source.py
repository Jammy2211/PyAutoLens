from autofit import conf
from autofit.core import non_linear as nl
from autolens.data import ccd
from autolens.data.array import mask as msk
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase
from autolens.pipeline import pipeline
from autolens.data.plotters import ccd_plotters

import os

# In chapter 2, we fitted a strong lens which included the contribution of light from the lens galaxy. We're going to
# fit this lens again (I promise, this is the last time!). However, now we're using pipelines, we can perform a 
# completely different (and significantly faster) analysis.

# Load up the PDFs from the previous tutorial -
# 'howtolens/chapter_2_lens_modeling/output/5_linking_phase_2/image/pdf_triangle.png'.

# This is a big triangle. As we fit models using more and more parameters, its only going to get bigger!

# As usual, you should notice some clear degeneracies between:

# 1) The size (effective_radius, R_l) and intensity (intensity, I_l) of light profiles.
# 2) The mass normalization (einstein_radius, /Theta_m) and ellipticity (axis_ratio, q_m) of mass profiles.

# This isn't surprising. You can produce similar looking galaxies by trading out intensity for size, and you can
# produce similar mass distributions by compensating for a loss in lens mass by making it a bit less elliptical.

# What do you notice about the contours between the lens galaxy's light-profile and its mass-profile / the source
# galaxy's light profile? Look again.

# That's right - they're not degenerate. The covariance between these sets of parameters is minimal. Again, this makes
# sense - why would fitting the lens's light (which is an elliptical blob of light) be degenerate with fitting the
# source's light (which is a ring of light)? They look nothing like one another!

# So, as a newly trained lens modeler, what does the lack of covariance between these parameters make you think?
# Hopefully, you're thinking, why would I even both fitting the lens and source galaxy simultaneously? Certainly not
# at the beginning of an analysis, when we just want to find the right regions of non-linear parameter space. This is
# what we're going to do in this tutorial, using a pipeline composed of a modest 3 phases:

# Phase 1) Fit the lens galaxy's light, ignoring the source.
# Phase 2) Fit the source galaxy's light, ignoring the lens.
# Phase 3) Fit both simultaneously, using these results to initialize our starting location in parameter space.

# From here on, we'll use the configs in the 'workspace/config' folder, which are the default configs used
# by all pipelines (e.g. not just this tutorial, but when you model your own images and lenses!).


# To set these up without docker, you need to uncomment and run the command below. If you are using Docker, you don't
# need to do anything so leave this uncommented!
path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))
conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

# We'll also put the output in 'workspace/output', which is where output goes for a normal analysis.

# This is true of all tutorials from now on, to get you used to using the code for modeling real images.

def simulate():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.simulate_as_gaussian(shape=(11, 11), sigma=0.1, pixel_scale=0.1)

    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(shape=(130, 130), pixel_scale=0.1, 
                                                                       psf_shape=(11, 11))

    lens_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, intensity=0.04,
                                                     effective_radius=0.5, sersic_index=3.5),
                           mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0,
                                                        einstein_radius=1.0),
                           shear=mp.ExternalShear(magnitude=0.05, phi=90.0))
    
    source_galaxy = g.Galaxy(light=lp.SphericalExponential(centre=(0.0, 0.0), intensity=0.2, effective_radius=0.2))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=image_plane_grid_stack)

    return ccd.CCDData.simulate(array=tracer.image_plane_image_for_simulation, pixel_scale=0.1,
                               exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# Now lets simulate the ccd data we'll fit, which as I said above, is the same image we saw in the previous chapter.
ccd_data = simulate()
ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

# A pipeline is a one long python function (this is why Jupyter notebooks arn't ideal). When we run it, this function
# 'makes' the pipeline, as you'll see in a moment.
def make_pipeline(pipeline_name):

    # The pipeline takes a name as input, which specifies the directory that it appears in the output folder.

    ### PHASE 1 ###

    # In chapter 2, we learnt how to mask data for a phase. For pipelines, we are probably going to want to change our
    # mask between phases. Afterall, the bigger the mask, the slower the run-time, so during the early phases of most
    # pipelines we're probably not too bothered about fitting all of the image. Aggresive masking (which removes lots
    # of image-pixels) is thus an appealing way to get things running fast.

    # In this phase, we're only interested in fitting the lens's light, so we'll mask out the source-galaxy entirely.
    # This'll give us a nice speed up and ensure the source's light doesn't impact our light-profile fit (albeit,
    # given the lack of covariance above, it wouldn't be disastrous either way).

    # We want a mask that is shaped like the source-galaxy. The shape of the source is an 'annulus' (e.g. a ring),
    # so we're going to use an annular masks. For example, if an annulus is specified between an inner radius of 0.5"
    # and outer radius of 2.0", all pixels in two rings between 0.5" and 2.0" are included in the analysis.

    # But wait, we actually want the opposite of this. We want a masks where the pixels between 0.5" and 2.0" are not
    # included! They're the pixels the source is actually located. Therefore, we're going to use an 'anti-annular
    # mask', where the inner and outer radii are the regions we omit from the analysis. This means we need to specify
    # a third ring of the masks, even further out, such that data at these exterior edges of the image are also masked.

    # We can change a mask using the 'mask_function', which basically returns the new masks we want to use (you can
    # actually use on phases by themselves like in the previous chapter).

    def mask_function(image):
        return msk.Mask.circular_anti_annular(image.shape, pixel_scale=image.pixel_scale, inner_radius_arcsec=0.5,
                                               outer_radius_arcsec=1.6, outer_radius_2_arcsec=2.5)

    # We next create the phase, using the same notation we learnt before (but noting the masks function is passed to
    # this phase ensuring the anti-annular masks above is used).

    phase1 = phase.LensPlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                  optimizer_class=nl.MultiNest, mask_function=mask_function,
                                  phase_name=pipeline_name + '/phase_1_lens_light_only')

    # We'll use the MultiNest black magic we covered in tutorial 7 of chapter 2 to get this phase to run fast.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.5

    ### PHASE 2 ###

    # In phase 2, we fit the source galaxy's light. Thus, we want to make 2 changes from the previous phase

    # 1) We want to fit the lens subtracted image calculated in phase 1, instead of the observed image.
    # 2) We want to mask the central regions of this image, where there are residuals due to the lens light subtraction.

    # We can use the mask function again, to modify the mask to an annulus. We'll use the same ring radii as before.

    def mask_function(img):
        return msk.Mask.circular_annular(img.shape, pixel_scale=img.pixel_scale, inner_radius_arcsec=0.5,
                                          outer_radius_arcsec=3.)

    # To modify an image, we call a new function, 'modify image'. This function behaves like the pass-priors functions
    # before, whereby we create a python 'class' in a Phase to set it up.  This ensures it has access to the pipeline's
    # 'previous_results' (which you may have noticed was in the the pass_priors functions as well, but we ignored it
    # thus far).

    # To setup the modified image, we take the observed image data and subtract-off the model image from the
    # previous phase, which, if you're keeping track, is an image of the lens galaxy. However, if we just used the
    # 'model_image' in the fit, this would only include pixels that were masked. We want to subtract the lens off the
    # entire image - fortunately, PyAutoLens automatically generates a 'unmasked_lens_plane_model_image' as well!

    class LensSubtractedPhase(phase.LensSourcePlanePhase):

        def modify_image(self, image, previous_results):
            phase_1_results = previous_results[0]
            return image - phase_1_results.most_likely_fit.unmasked_lens_plane_model_image

    # The function above demonstrates the most important thing about pipelines - that every phase has access to the
    # results of all previous phases. This means we can feed information through the pipeline and therefore use the
    # results of previous phases to setup new phases. We'll see this again in phase 3.

    # We setup phase 2 as per usual. Note that we don't need to pass the modify image function.

    phase2 = LensSubtractedPhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                 source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                 optimizer_class=nl.MultiNest, mask_function=mask_function,
                                 phase_name=pipeline_name + '/phase_2_source_only')

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    ### PHASE 3 ###

    # Finally, in phase 3, we want to fit the lens and source simultaneously. Whilst we made a point of them not being
    # covariant above, there will be some level of covariance that will, at the very least, impact the errors we infer.
    # Furthermore, if we did go on to fit a more complex model (say, a decomposed light and dark matter model - yeah,
    # you can do that in PyAutoLens!), fitting the lens's light and mass simultaneously would be crucial.

    # We'll use the 'pass_priors' function that we all know and love to do this. However, we're going to use the
    # 'previous_results' argument that, in chapter 2, we ignored. This stores the results of the lens model of
    # phase 1 and 2, meaning we can use it to initialize phase 3's priors!

    class LensSourcePhase(phase.LensSourcePlanePhase):

        def pass_priors(self, previous_results):

            # The previous results is a 'list' in python. The zeroth index entry of the list maps to the results of
            # phase 1, the first entry to phase 2, and so on.

            phase_1_results = previous_results[0]
            phase_2_results = previous_results[1]

            # To link two priors together, we invoke the 'variable' attribute of the previous results. By invoking
            # 'variable', this means that:

            # 1) The parameter will be a free-parameter fitted for by the non-linear search.
            # 2) It will use a GaussianPrior based on the previous results as its initialization (we'll cover how this
            #    Gaussian is setup in tutorial 4, for now just imagine it links the results in a sensible way).

            # We can simple link every source galaxy parameter to its phase 2 inferred value, as follows

            self.source_galaxies.source.light.centre_0 = phase_2_results.variable.source.light.centre_0
            self.source_galaxies.source.light.centre_1 = phase_2_results.variable.source.light.centre_1
            self.source_galaxies.source.light.axis_ratio = phase_2_results.variable.source.light.axis_ratio
            self.source_galaxies.source.light.phi = phase_2_results.variable.source.light.phi
            self.source_galaxies.source.light.intensity = phase_2_results.variable.source.light.intensity
            self.source_galaxies.source.light.effective_radius = \
                phase_2_results.variable.source.light.effective_radius
            self.source_galaxies.source.light.sersic_index = phase_2_results.variable.source.light.sersic_index

            # However, listing every parameter like this is ugly, and is combersome if we have a lot of parameters.

            # If, like in the above example, you are making all of the parameters of a lens or source galaxy variable,
            # you can simply set the source galaxy equal to one another, without specifying each parameter of every
            # light and mass profile.

            self.source_galaxies.source = phase_2_results.variable.source # This is identical to lines 196-203 above.

            # For the lens galaxies, we have a slightly weird circumstance where the light profiles requires the
            # results of phase 1 and the mass profile the results of phase 2. When passing these as a 'variable', we
            # can split them as follows

            self.lens_galaxies.lens.light = phase_1_results.variable.lens.light
            self.lens_galaxies.lens.mass = phase_2_results.variable.lens.mass

    phase3 = LensSourcePhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                    mass=mp.EllipticalIsothermal)),
                             source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                             optimizer_class=nl.MultiNest, phase_name=pipeline_name + '/phase_3_both')

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3)


pipeline_lens_and_source = make_pipeline(pipeline_name='howtolens/c3_t1_lens_and_source')
pipeline_lens_and_source.run(data=ccd_data)

# And there we have it, a pipeline that breaks the analysis of the lens and source galaxy into 3 simple phases. This
# approach is signifcantly faster than fitting the lens and source simultaneously from the beginning. Instead of asking
# you questions at the end of this chapter's tutorials, I'm gonna give Q&A's - this'll hopefully get you thinking about
# how to approach pipeline writing.

# 1) Can this pipeline really be generalized to any lens? Surely the radii of the masks in phase 1 and 2 depend on the
#    lens and source galaxies?
#
#    Whilst this is true, we've chosen values of masks radii above that are 'excessive' and mask out a lot more of the
#    image than just the source (which, in terms of run-time, is desirable). Thus, provided you know the Einstein
#    radius distribution of your lens sample, you can choose masks radii that will masks out every source in your sample
#    adequately (and even if some of the source is still there, who cares? The fit to the lens galaxy will be okay).

#    Furthermore, we discussed in chapter 2, tutorial 8 the ability to input custom masks to a phase. You can also input
#    custom masks to a pipeline, which are used by default if a mask_function is not supplied. I haven't included a
#    howtolens tutorial on this, but if you navigate to the workspace/pipelines/examples/ folder you'll find an example
#    pipeline describingn how to do this ('mask_and_positions.py').

# 2) What if my source galaxy isn't a ring of light? Surely my Annulus masks won't match it?
#
#    Just use the annulus anyway! Yeah, you'll masks out lots of image pixels with no source light, but remember, at
#    the beginning of the pipeline, *we don't care*. In phase 3, we'll use a large circular masks and do the fit
#    properly.
