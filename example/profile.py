from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline as pl
from autolens.autopipe import non_linear as nl
from autolens.analysis import galaxy_prior as gp
from autolens.pixelization import pixelization
from autolens.imaging import mask as msk
from autolens.imaging import image as im
from autolens.profiles import light_profiles, mass_profiles

# Load an image from the 'basic' folder. It is assumed that this folder contains image.fits, noise.fits and psf.fits.
img = im.load('basic', pixel_scale=0.05)

# In the first phase we attempt to fit the lens light with an EllipticalSersicLP.
phase1 = ph.LensProfilePhase(lens_galaxy=gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersicLP),
                             lens_satellite_1=gp.GalaxyPrior(elliptical_sersic=light_profiles.EllipticalSersicLP)
                          optimizer_class=nl.MultiNest)


# In the second phase we remove the lens light found in the first phase and try to fit just the source. To do this we
# extend LensMassAndSourceProfilePhase class and override two methods.
class LensAndSubtractedPlanePhase(ph.LensMassAndSourceProfilePhase):
    # The modify image method provides a way for us to modify the image before a phase starts.
    def modify_image(self, image, previous_results):
        # The image is the original image we are trying to fit. Previous results is a list of results from previous
        # phases. We access the result from the last phase by calling previous_results.last. We take the image of the
        # lens galaxy from the last phase from the image.
        return image - previous_results.last.lens_plane_blurred_image_plane_image

    # The pass prior method provides us with a way to set variables and constants in this phase using those from a
    # previous phase.
    def pass_priors(self, previous_results):
        # Here we set the centre of the mass profile of our lens galaxy to a prior provided by the light profile of the
        # lens galaxy in the previous phase. This means that the centre is still variable, but constrained to a set of
        # likely values in parameter space.
        self.lens_galaxies.sie.centre = previous_results.last.variable.lens_galaxy.elliptical_sersic.centre


# A mask function determines which parts of the image should be masked out in analysis. By default the mask is a disc
# with a radius of 3 arc seconds. Here we are only interested in light from the source galaxy so we define an annular
# mask.
def annular_mask_function(image):
    return msk.Mask.annular(image.shape_arc_seconds, pixel_scale=image.pixel_scale, inner_radius=0.4,
                            outer_radius=3.)


# We create the second phase. It's an instance of the phase class we created above with the custom mask function passed
# in. We have a lens galaxy with an SIE mass profile and a source galaxy with an Elliptical Sersic light profile. The
# centre of the mass profile we pass in here will be constrained by the pass_priors function defined above.
phase2 = LensAndSubtractedPlanePhase(source_galaxy=gp.GalaxyPrior(pixelization=pixelization.Rectangular),
                                     optimizer_class=nl.MultiNest)


# In the third phase we try fitting both lens and source light together. We use priors determined by both the previous
# phases to constrain parameter space search.
class CombinedPlanePhaseAnd(ph.LensMassAndSourceProfilePhase):
    # We can take priors from both of the previous results.
    def pass_priors(self, previous_results):
        # The lens galaxy's light profile is constrained using results from the first phase whilst its mass profile
        # depends on results from the second phase.
        self.lens_galaxy = gp.GalaxyPrior(
            elliptical_sersic=previous_results.first.variable.lens_galaxy.elliptical_sersic,
            sie=previous_results.last.variable.lens_galaxy.sie)
        # The source galaxy is constrained using the second phase.
        self.source_galaxy = previous_results.last.variable.source_galaxy


# We define the lens and source galaxies explicitly in pass_priors so there's no need to pass them in when we make the
# phase3 instance
phase3 = CombinedPlanePhaseAnd(optimizer_class=nl.MultiNest)

# Phase3h is used to fit hyper galaxies. These objects are used to scale noise and prevent over fitting.
phase3h = ph.SourceLensAndHyperGalaxyPlanePhase()


# The final phase tries to fit the whole system again.
class CombinedPlanePhase2And(ph.LensMassAndSourceProfilePhase):
    # We take priors for the galaxies from phase 3 and set their hyper galaxies from phase 3h.
    def pass_priors(self, previous_results):
        phase_3_results = previous_results[2]
        # Note that a result has a variable attribute with priors...
        self.lens_galaxy = phase_3_results.variable.lens_galaxy
        self.source_galaxy = phase_3_results.variable.source_galaxy
        # ...and a constant attribute with fixed values. The fixed values are the best fit from the phase in question.
        # We fix the 'hyper_galaxy' property of our lens and source galaxies here.
        self.lens_galaxy.hyper_galaxy = previous_results.last.constant.lens_galaxy.hyper_galaxy
        self.source_galaxy.hyper_galaxy = previous_results.last.constant.source_galaxy.hyper_galaxy


# We create phase4. Once again, both lens and source galaxies are defined in pass_priors so there's no need to pass
# anything into the constructor.
phase4 = CombinedPlanePhase2And(optimizer_class=nl.MultiNest)

# We put all the phases together in a pipeline and give it a name.
pipeline = pl.Pipeline("profile_pipeline", phase1, phase2, phase3, phase3h, phase4)

# The pipeline is run on an image.
results = pipeline.run(img)

# Let's print the results.
for result in results:
    print(result)
