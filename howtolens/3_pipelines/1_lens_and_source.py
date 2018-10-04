from autolens.pipeline import phase
from autolens.pipeline import pipeline
from autolens.autofit import non_linear as nl
from autolens.imaging import image as im
from autolens.imaging import mask
from autolens.lensing import galaxy_model as gm
from autolens.profiles import light_profiles as lp
from autolens.profiles import mass_profiles as mp

# In chapter 2, we fitted a strong lens which included the contribution of light from the lens galaxy. We're going to
# fit this lens again (I promise, this is the last time). However, now we're approaching lens modeling with pipelines,
# we can perform a completely different (and significantly faster) analysis.

# Load up the PDFs from the previous tutorial - 'output/howtolens/5_linking_phase_2/optimizer/chains/pdfs/Triangle.png'.

# This is a big triangle. As we fit models using more and more parameters, its only going to get bigger!

# As usual, you should notice some clear degeneracies between:

# 1) The size (effective_radius, R_l) and intensity (intensity, I_l) of light profiles.
# 2) The mass normalization (einstein_radius, Theta_m) and ellipticity (axis_ratio, q_m) of mass profiles.

# This isn't surprising. You can produce similar looking galaxies by trading out intensity for size, and you can
# produce similar mass distributions by compensating for lens mass by making it a bit less elliptical.

# What do you notice about the contours between the lens galaxy's light-profile and its mass-profile / the source
# galaxy's light profile? Look again.

# That's right - they're not degenerate. The covariance between these sets of parameters is minimal. Again, this makes
# sense - why would fitting the lens's light (which is an elliptical blob of light) be denegerate with fitting the
# source's light,(which is a ring of light)? They look nothing like one another!

# So, as a newly trained lens modeler, what does the lack of covariance between these parameters make you think?
# Hopefully, you're thinking, why would I even both fitting the lens and source galaxy simultaneously? Certainly not
# at the beginning of an analysis, when we just want to find the right regions of non-linear parameter space. This is
# what we're gonna do in this tutorial, using a pipeline composed of a modest 3 phases:

# Phase 1 - Fit the lens galaxy's light, ignoring the source.
# Phase 2 - Fit the source galaxy's light, ignoring the lens.
# Phase 3 - Fit both simultaneously, using these results to start in the right regions of parameter space.

# A pipeline is a large function (this is why Jupyter notebooks arn't ideal). This function, when we run it, 'makes'
# the pipeline, as you'll see in a moment.
def make_pipeline():

    # To begin, we name our pipeline, which will specify the directory that it appears in the output folder.
    pipeline_name = 'howtolens/lens_and_source'  # Give the pipeline a name.

    # The first thing we can do in pipelines is change the mask of each phase. If we're only interested in fitting the
    # lens's light this phase, we may as well mask out the source-galaxy. This will speed up the analysis and ensure
    # that the source's light doesn't impact our light-profile fit (albeit, given the lack of covariance above, we
    # know this wouldn't be too bad anyway).

    # We want a mask that is shaped like the source-galaxy. The shape of the source is an 'annulus' (e.g. a donut),
    # so we're going to use an annular mask. For example, if an annulus is specified between an inner radius of 0.5"
    # and outer radius of 2.0", all pixels in the rings between 0.5" and 2.0" would be included in the analysis.

    # But wait, we actually want the opposite of this. We want a mask where the pixels between 0.5" and 2.0" are not
    # included! They're the pixels the source is actually located. Therefore, we're going to use an 'anti-annular
    # mask', where the inner and outer radii are the regions we omit from the analysis. This means we need to specify
    # a third radius of the mask, even further out, such that data at the exterior edges of the image begins to be
    # masked again.

    #To change a mask, we call the mask function
    def mask_function(img):
        return mask.Mask.anti_annular(img.shape, pixel_scale=img.pixel_scale, inner_radius_arcsec=0.5,
                                      outer_radius_arcsec=1.6, outer_radius_2_arcsec=3.0)

    # We now create the phase, using the same notation we learnt before (but noting the mask function above has now
    # also been passed to this phase, meaning the mask above will be used.
    phase1 = phase.LensPlanePhase(lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic)],
                                  optimizer_class=nl.MultiNest, mask_function=mask_function,
                                  phase_name=pipeline_name+'/phase_1_lens_light_only')

    # In phase 2, we fit the source galaxy's light. Thus, we want to make 2 changes from the previous phase:

    # 1) We want to fit the lens subtracted image computed in phase 1, instead of the observed image.
    # 2) We want to mask the central regions of this image, where there are residuals due to the lens light subtraction.

    # We can use the mask function again, to modify the mask to an annulus. And we can call a new function,
    # 'modify image', to modify the image we fit to the lens subtracted image. Note that, the modify image function
    # behaves like the pass-priors functions before, whereby we create a python 'class' in a Phase to set it up.

    # Its worth stressing something about the modify image function (which you may have noticed about the pass_priors
    # function previously). We're using the *previous_results* of the pipeline (specifically the results of phase 1)
    # to change the image.
#
    # This highlights an extremely important point about pipelines in PyAutoLens - that they allow us to feed
    # information through the pipelines from all of the previous phases, Cruciallyy, this means that their results can
    # be used in subsequent phases. We're going to see this again in phase 3.
    class LensSubtractedPhase(phase.LensSourcePlanePhase):

        def modify_image(self, image, previous_results):
            phase_1_results = previous_results[0]
            return image - phase_1_results.fit.model_image

    def mask_function(img):
        return mask.Mask.annular(img.shape, pixel_scale=img.pixel_scale, inner_radius_arcsec=0.5,
                                 outer_radius_arcsec=3.)

    # We setup phase 2 as per usual. Note that we don't need to pass the modify image function.
    phase2 = LensSubtractedPhase(lens_galaxies=[gm.GalaxyModel(mass=mp.EllipticalIsothermal)],
                                 source_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic)],
                                 optimizer_class=nl.MultiNest, mask_function=mask_function,
                                 phase_name=pipeline_name+'/phase_2_source_only')

    # Finally, in phase 3, we want to fit the lens and source simulateously. Whilst we made a point of them not being
    # covariant above, there probably is some level of covariance that will impact the errors we infer on the parameters.

    # Furthermore, if we did go on to fit a more complex model (say, a decomposed light and dark matter model - yeah,
    # you can do that in PyAutoLens!), setting up a phase in this way would be crucial. In you don't believe me,
    # maybe you will in a few tutorials time.

    # We use the 'pass_priors' function that we are all now familiar with. However, we're goiong to use that
    # 'previous_results' argument that, up to now, we've pretty much ignored. This stores the results of the phase's 1
    # and 2, meaning that we can use it to initialize phase 3's priors! Theres a bit of new syntax below, but you
    # should follow quite nicely whats going on.
    #
    # which allows us to
    # map the previous_results of the lens and source galaxies to the galaxies in this phase.
    # The term 'variable' signifies that when we map these results to setup the GalaxyModel, we want the parameters

    # To still be treated as free parameters that are varied during the fit.

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
            #    Gaussian is setup in a later tutorial).

            self.source_galaxies[0].light.centre_0 = phase_2_results.variable.source_galaxies.light.centre_0
            self.source_galaxies[0].light.centre_1 = phase_2_results.variable.source_galaxies.light.centre_1
            self.source_galaxies[0].light.axis_ratio = phase_2_results.variable.source_galaxies.light.axis_ratio
            self.source_galaxies[0].light.phi = phase_2_results.variable.source_galaxies.light.phi
            self.source_galaxies[0].light.intensity = phase_2_results.variable.source_galaxies.light.intensity
            self.source_galaxies[0].light.effective_radius = \
                phase_2_results.variable.source_galaxies.light.effective_radius
            self.source_galaxies[0].light.sersic_index = phase_2_results.variable.source_galaxies.light.sersic_index

            # Listing every parameter like this is a bit ugly. If, like in the above example, you are making all
            # parameters of a lens or source galaxy variable, you can simply set them equal to one another without
            # specifying the light / mass profiles
            self.source_galaxies = phase_2_results.variable.source_galaxies

            # For the lens galaxies, we have a slightly weird circumstance where the light profiles requires the results
            # of phase 1, and the mass profile of phase 2. When passing these as 'variables, we can split them up
            # using a GalaxyModel (as opposed to listing every indvividual parameter).
            self.lens_galaxies[0] = gm.GalaxyModel(light=phase_1_results.variable.lens_galaxies[0].light,
                                                   mass=phase_2_results.variable.lens_galaxies[0].mass)

    phase3 = LensSourcePhase(lens_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic,
                                                           mass=mp.EllipticalIsothermal)],
                              source_galaxies=[gm.GalaxyModel(light=lp.EllipticalSersic)],
                              optimizer_class=nl.MultiNest, phase_name='ph3')

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3)

# This is the script which loads the image, makes the pipeline and runs it.
path = '/home/jammy/PyCharm/Projects/AutoLens/howtolens/2_lens_modeling'
image = im.load_imaging_from_path(image_path=path + '/data/3_realism_and_complexity_image.fits',
                                  noise_map_path=path+'/data/3_realism_and_complexity_noise_map.fits',
                                  psf_path=path + '/data/3_realism_and_complexity_psf.fits', pixel_scale=0.1)

pipeline_lens_and_source = make_pipeline()
pipeline_lens_and_source.run(image=image)


# And there we have it, a pipeline that breaks the analysis of the lens and source galaxy into a set of phases. This
# approach is signnifcantly faster than fitting all of their associated parameters at once.