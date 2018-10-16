from howtolens.simulations import pipelines as simulation

from autolens import conf
from autolens.autofit import non_linear as nl
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.lensing import galaxy_model as gm
from autolens.pipeline import phase
from autolens.pipeline import pipeline
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp

# In chapter 2, we fitted a strong lens which included the contribution of light from the lens galaxy. We're going to
# fit this lens again (I promise, this is the last time!). However, now we're approaching lens modeling with runners,
# we can perform a completely different (and significantly faster) analysis.

# Load up the PDFs from the previous tutorial -
# 'output/howtolens/2_lens_modeling/5_linking_phase_2/images/pdf_triangle.png'.

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

# Phase 1 - Fit the lens galaxy's light, ignoring the source.
# Phase 2 - Fit the source galaxy's light, ignoring the lens.
# Phase 3 - Fit both simultaneously, using these results to initialize our starting location in parameter space.

# First, we need to setup the config. No, I'm not preloading the results with this - I'm just changing the output
# to the AutoLens/howtolens/output directory, to keep everything tidy and in one place.

conf.instance = conf.Config(config_path=conf.CONFIG_PATH, output_path="../output")

# Now lets simulate hte image we'll fit, which as I said above, is the same image we saw in the previous chapter.
simulation.pipeline_lens_and_soure_image()

# Lets load the image before writing our pipeline.
image = im.load_imaging_from_path(image_path='data/1_lens_and_source/image.fits',
                                  noise_map_path='data/1_lens_and_source/noise_map.fits',
                                  psf_path='data/1_lens_and_source/psf.fits', pixel_scale=0.1)


# A pipeline is a one long python function (this is why Jupyter notebooks arn't ideal). When we run it, this function
# 'makes' the pipeline, as you'll see in a moment.
def make_pipeline():
    # To begin, we name our pipeline, which will specify the directory that it appears in the output folder.
    pipeline_name = '3_pipelines/1_lens_and_source'

    # Its been a long time since we thought about masks - but in runners they're a pretty important. The bigger the
    # mask, the slower the run-time. In the early phases of most runners, we're not too bothered about fitting the
    # image perfectly and aggresive masking (removing lots of image-pixels) is a good way to get things running fast.

    # In this phase, we're only interested in fitting the lens's light, so we'll mask out the source-galaxy entirely.
    # This'll give us a nice speed up and ensure the source's light doesn't impact our light-profile fit (albeit,
    # given the lack of covariance above, it wouldn't be disastrous either way).

    # We want a mask that is shaped like the source-galaxy. The shape of the source is an 'annulus' (e.g. a ring),
    # so we're going to use an annular mask. For example, if an annulus is specified between an inner radius of 0.5"
    # and outer radius of 2.0", all pixels in two rings between 0.5" and 2.0" are included in the analysis.

    # But wait, we actually want the opposite of this. We want a mask where the pixels between 0.5" and 2.0" are not
    # included! They're the pixels the source is actually located. Therefore, we're going to use an 'anti-annular
    # mask', where the inner and outer radii are the regions we omit from the analysis. This means we need to specify
    # a third rung of the mask, even further out, such that data at the exterior edges of the image is also masked.

    # We can change a mask using the 'mask_function' which basically returns the new mask we want to use (you can
    # actually use on phases by themselves like in the previous chapter).
    def mask_function(img):
        return mask.Mask.anti_annular(img.shape, pixel_scale=img.pixel_scale, inner_radius_arcsec=0.5,
                                      outer_radius_arcsec=1.6, outer_radius_2_arcsec=3.0)

    # Create the phase, using the same notation we learnt before (but noting the mask function is passed to this phase,
    # ensuring the anti-annular mask above is used).
    phase1 = phase.LensPlanePhase(lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic)],
                                  optimizer_class=nl.MultiNest, mask_function=mask_function,
                                  phase_name=pipeline_name + '/phase_1_lens_light_only')

    # At this point, you might want to look at the 'output/3_pipelines/1_lens_and_source/phase_1_lens_light_only'
    # output folder. There'll you'll find the model results and images, so you can be sure this phase runs as expected!

    # In phase 2, we fit the source galaxy's light. Thus, we want to make 2 changes from the previous phase

    # 1) We want to fit the lens subtracted image calculated in phase 1, instead of the observed image.
    # 2) We want to mask the central regions of this image, where there are residuals due to the lens light subtraction.

    # We can use the mask function again, to modify the mask to an annulus. We'll use the same ring radii as before,
    # but this isn't necessary.
    def mask_function(img):
        return mask.Mask.annular(img.shape, pixel_scale=img.pixel_scale, inner_radius_arcsec=0.5,
                                 outer_radius_arcsec=3.)

    # To modify an image, we call a new function, 'modify image'. This function behaves like the pass-priors functions
    # before, whereby we create a python 'class' in a Phase to set it up.  This ensures it has access to the pipeline's
    # 'previous_results' (which you may have noticed was in the the pass_priors functions as well, but we ignored it
    # thus far).

    # To setup the modified image, we take the observed image data ('image') and subtract-off the model image from the
    # previous phase, which, if you're keeping track, is an image of the lens galaxy. However, if we just used the
    # 'model_image' in the fit, this would only include pixels that were masked. We want to subtract the lens off the
    # entire image - fortunately, PyAutoLens automatically generates a 'unmasked_model_image' as well!

    class LensSubtractedPhase(phase.LensSourcePlanePhase):

        def modify_image(self, image, previous_results):
            phase_1_results = previous_results[0]
            return image - phase_1_results.fit.unmasked_model_profile_image

    # The function above demonstrates the most important thing about runners - that every phase has access to the
    # results of all previous phases. This means we can feed information through the pipeline and therefore use the
    # results of previous phases to setup new phases. We'll see this again in phase 3.

    # We setup phase 2 as per usual. Note that we don't need to pass the modify image function.
    phase2 = LensSubtractedPhase(lens_galaxies=[gm.GalaxyModel(mass=mp.EllipticalIsothermal)],
                                 source_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic)],
                                 optimizer_class=nl.MultiNest, mask_function=mask_function,
                                 phase_name=pipeline_name + '/phase_2_source_only')

    # Finally, in phase 3, we want to fit the lens and source simultaneously. Whilst we made a point of them not being
    # covariant above, there will be some level of covariance that will, at the very least, impact the errors we infer.
    # Furthermore, if we did go on to fit a more complex model (say, a decomposed light and dark matter model - yeah,
    # you can do that in PyAutoLens!), setting up a phase in this way would be crucial.

    # We'll use the 'pass_priors' function that we are know and love now. However, we're going to use the
    # 'previous_results' argument that, up to now, we've ignored. This stores the results of the lens model results of
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
            #    Gaussian is setup later).

            self.source_galaxies[0].light.centre_0 = phase_2_results.variable.source_galaxies[0].light.centre_0
            self.source_galaxies[0].light.centre_1 = phase_2_results.variable.source_galaxies[0].light.centre_1
            self.source_galaxies[0].light.axis_ratio = phase_2_results.variable.source_galaxies[0].light.axis_ratio
            self.source_galaxies[0].light.phi = phase_2_results.variable.source_galaxies[0].light.phi
            self.source_galaxies[0].light.intensity = phase_2_results.variable.source_galaxies[0].light.intensity
            self.source_galaxies[0].light.effective_radius = \
                phase_2_results.variable.source_galaxies[0].light.effective_radius
            self.source_galaxies[0].light.sersic_index = phase_2_results.variable.source_galaxies[0].light.sersic_index

            # Listing every parameter like this is ugly, and would get unweildy if we had a lot of parameters. If,
            # like in the above example, you are making all parameters of a lens or source galaxy variable, you can
            # simply set the source_galaxies equal to one another without specifying the light / mass profiles
            self.source_galaxies = phase_2_results.variable.source_galaxies

            # For the lens galaxies, we have a slightly weird circumstance where the light profiles requires the
            # results of phase 1 and the mass profile the results of phase 2. When passing these as a 'variable', we
            # can split them using a GalaxyModel (we could list every individual parameter, but that'd be pretty long).
            self.lens_galaxies[0] = gm.GalaxyModel(light=phase_1_results.variable.lens_galaxies[0].light,
                                                   mass=phase_2_results.variable.lens_galaxies[0].mass)

    phase3 = LensSourcePhase(lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic,
                                                           mass=mp.EllipticalIsothermal)],
                             source_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic)],
                             optimizer_class=nl.MultiNest, phase_name='/phase_3_both')

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3)


pipeline_lens_and_source = make_pipeline()
pipeline_lens_and_source.run(image=image)

# And there we have it, a pipeline that breaks the analysis of the lens and source galaxy into a set of phases. This
# approach is signifcantly faster than fitting everything at once. Instead of asking you questions at the end of
# this chapter's tutorials, I'm gonna give Q&A's - this'll hopefully get you thinking about how we should approach
# pipeline writing.

# 1) Can this pipeline really be generalized to any lens? Surely the radii of the masks in phase 1 and 2 depend on the
#    lens and source galaxies?
#
#    Whilst this is true, we've chosen values of mask radii above that are 'excessive' that mask out a lot more of the
#    image than just the source (which, in terms of run-time, is desirable). Thus, provided you know the Einstein
#    radius distribution of your lens sample, you can choose mask radii that will mask out every source in your sample
#    adequately (and even if some of the source is still there, who cares? The fit to the lens galaxy will be okay).

# 2) What if my source galaxy isn't a ring of light? Surely my Annulus mask won't match it?
#
#    Just use the annulus anyway! Yeah, you'll mask out lots of image pixels with no source light, but remember, at
#    the beginning of the pipeline, *we don't care*. In phase 3, we'll use a large circular mask and do the fit
#    properly.
