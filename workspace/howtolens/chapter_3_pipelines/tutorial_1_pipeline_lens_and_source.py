from autofit.optimize import non_linear as nl
from autolens.data.array import mask as msk
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase
from autolens.pipeline import pipeline

# I recommend that any pipeline you write begins at the top with a comment like this, which describes te pipeline and
# gives a phase-by-phase description of what it does.

# In this pipeline, we'll perform a basic analysis which fits a source galaxy using a parametric light profile and a
# lens galaxy where its light is included and fitted, using three phases:

# Phase 1) Fit the lens galaxy's light using an elliptical Sersic light profile.

# Phase 2) Use this lens subtracted image to fit the lens galaxy's mass (SIE) and source galaxy's light (Sersic).

# Phase 3) Fit the lens's light, mass and source's light simultaneously using priors initialized from the above 2 phases.

def make_pipeline(pipeline_path=''):

    # Pipelines takes a path as input, which in conjunction with the pipeline name specify the directory structure of
    # its results in the output folder. In pipeline runners we'll pass the pipeline path
    # 'howtolens/c3_t1_lens_and_source', which means the results of this pipeline will go to the folder
    # 'output/howtolens/c3_t1_lens_and_source/pipeline_light_and_source'.

    # By default, the pipeline path is an empty string, such that without a pipeline path the results go to the output
    # directory 'output/pipeline_name', which in this case would be 'output/pipeline_light_and_source'.

    # In the example pipelines supplied with PyAutoLens in the workspace/pipeline/examples folder, you'll see that
    # we pass the name of our strong lens data to the pipeline path. This allows us to fit a large sample of lenses
    # using one pipeline and store all of their results in an ordered directory structure.

    pipeline_name = 'pipeline_light_and_source'
    pipeline_path = pipeline_path + pipeline_name

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
        return msk.Mask.circular_anti_annular(shape=image.shape, pixel_scale=image.pixel_scale, inner_radius_arcsec=0.5,
                                              outer_radius_arcsec=1.6, outer_radius_2_arcsec=2.5)

    # We next create the phase, using the same notation we learnt before (but noting the masks function is passed to
    # this phase ensuring the anti-annular masks above is used).

    phase1 = phase.LensPlanePhase(lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                  optimizer_class=nl.MultiNest, mask_function=mask_function,
                                  phase_name=pipeline_path + '/phase_1_lens_light_only')

    # We'll use the MultiNest black magic we covered in tutorial 7 of chapter 2 to get this phase to run fast.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.5

    ### PHASE 2 ###

    # In phase 2, we fit the source galaxy's light. Thus, we want to make 2 changes from the previous phase

    # 1) We want to fit the lens subtracted image calculated in phase 1, instead of the observed image.
    # 2) We want to mask the central regions of this image, where there are residuals due to the lens light subtraction.

    # We can use the mask function again, to modify the mask to an annulus. We'll use the same ring radii as before.

    def mask_function(image):
        return msk.Mask.circular_annular(shape=image.shape, pixel_scale=image.pixel_scale, inner_radius_arcsec=0.5,
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
            return image - phase_1_results.unmasked_lens_plane_model_image

    # The function above demonstrates the most important thing about pipelines - that every phase has access to the
    # results of all previous phases. This means we can feed information through the pipeline and therefore use the
    # results of previous phases to setup new phases. We'll see this again in phase 3.

    # We setup phase 2 as per usual. Note that we don't need to pass the modify image function.

    phase2 = LensSubtractedPhase(lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                 source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                 optimizer_class=nl.MultiNest, mask_function=mask_function,
                                 phase_name=pipeline_path + '/phase_2_source_only')

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
            self.source_galaxies.source.light.effective_radius = phase_2_results.variable.source.light.effective_radius
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
                             optimizer_class=nl.MultiNest, phase_name=pipeline_path + '/phase_3_both')

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_path, phase1, phase2, phase3)