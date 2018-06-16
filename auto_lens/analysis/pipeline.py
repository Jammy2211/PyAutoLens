from auto_lens.analysis import analysis
from auto_lens.analysis import galaxy_prior
from auto_lens.profiles import light_profiles, mass_profiles
from auto_lens.analysis import model_mapper as mm
from auto_lens import instrumentation as inst
from auto_lens.analysis import non_linear
from auto_lens.pixelization import pixelization

# Defines how wide gaussian prior should be
SIGMA_LIMIT = 3


def source_only_pipeline(image, mask, instrumentation):
    """
    Pipeline 1:

    PURPOSE - Fit a source-only image (i.e. no lens light component)

    PREPROCESSING:

    - Mark the brightest regions / multiple images of the source.
    - Draw a circle tracing the source (Einstein Radius / centre)
    - Draw circle / ellipse for the mask.

    NOTES:

    Image: Observed image used throughout.
    Mask: Assume a large mask (e.g. 2") throughout - this value could be chosen in preprocessing.
    """

    results = []

    """
    1) Mass: SIE+Shear
       Source: Sersic
       NLO: LM
    """
    optimizer_1 = non_linear.LevenbergMarquardt()

    source_galaxy_prior = galaxy_prior.GalaxyPrior(light_profile=light_profiles.EllipticalSersic)
    lens_galaxy_prior = galaxy_prior.GalaxyPrior(spherical_mass_profile=mass_profiles.SphericalIsothermal,
                                                 shear_mass_profile=mass_profiles.ExternalShear)

    optimizer_1.add("source_galaxies", [source_galaxy_prior])
    optimizer_1.add("lens_galaxies", [lens_galaxy_prior])

    result_1 = analysis.analyse(image, mask, optimizer_1, instrumentation=instrumentation)

    results.append(result_1)

    """
    2) Mass: SIE+Shear (priors from phase 1)
       Source: 'smooth' pixelization (include regularization parameter(s) in the model)
       NLO: LM
    """
    optimizer_2 = non_linear.LevenbergMarquardt()

    lens_galaxy_prior = galaxy_prior.GalaxyPrior(spherical_mass_profile=mass_profiles.SphericalIsothermal,
                                                 shear_mass_profile=mass_profiles.ExternalShear)
    source_galaxy_prior = galaxy_prior.GalaxyPrior(pixelization=pixelization.SquarePixelization)

    optimizer_2.add("lens_galaxies", [lens_galaxy_prior])
    optimizer_2.add("source_galaxies", [source_galaxy_prior])

    lens_galaxy_prior.spherical_mass_profile = result_1.priors.spherical_mass_profile
    lens_galaxy_prior.shear_mass_profile = result_1.priors.shear_mass_profile

    result_2 = analysis.analyse(image, mask, optimizer_2)

    results.append(result_2)

    """
    2H) Hyper-parameters: All included in model (most priors broad and uniform, but use previous phase regularization as well)
        Mass: SIE+Shear (Fixed to highest likelihood model from phase 2)
        Source: 'noisy' pixelization
        NLO: MN
    """
    optimizer_2h = non_linear.MultiNest()

    source_galaxy_prior = galaxy_prior.GalaxyPrior(pixelization=pixelization.VoronoiPixelization)

    optimizer_2h.add("source_galaxy", source_galaxy_prior)
    optimizer_2h.add("instrumentation", inst.Instrumentation)

    source_galaxy_prior.pixelization.regularization = result_2.priors.pixelization.regularization

    result_2h = analysis.analyse(image, mask, optimizer_2h, lens_galaxies=result_2.lens_galaxies)

    results.append(result_2h)

    """
    a) Mass: SPLE+Shear (priors from Init phase 2)
       Source: 'noisy' pixelization (Fixed to init 2H hyper-parameters)
    """
    optimizer_a = non_linear.MultiNest()

    lens_galaxy_prior = galaxy_prior.GalaxyPrior(spherical_power_law_mass_profile=mass_profiles.SphericalPowerLaw,
                                                 shear_mass_profile=mass_profiles.ExternalShear)

    optimizer_a.add("lens_galaxy", lens_galaxy_prior)

    lens_galaxy_prior.shear_mass_profile = result_2.prior.shear_mass_profile
    lens_galaxy_prior.spherical_power_law_mass_profile.centre = result_2.prior.spherical_mass_profile.centre

    result_a = analysis.analyse(image, mask, optimizer_a, instrumentation=result_2h.instrumentation,
                                pixelization=result_2h.pixelization)
    results.append(result_a)

    return results
