from auto_lens.analysis import analysis as an
from auto_lens.analysis import galaxy_prior
from auto_lens.analysis import model_mapper
from auto_lens.profiles import light_profiles, mass_profiles
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

    # Create an array in which to store results
    results = []

    # Create an analysis object for the image and mask
    analysis = an.Analysis(image, mask)

    """
    1) Mass: SIE+Shear
       Source: Sersic
       NLO: LM
    """
    # Create an optimizer
    optimizer_1 = non_linear.DownhillSimplex()

    # Define galaxy priors
    source_galaxy_prior = galaxy_prior.GalaxyPrior(light_profile=light_profiles.EllipticalSersic)
    lens_galaxy_prior = galaxy_prior.GalaxyPrior(spherical_mass_profile=mass_profiles.SphericalIsothermal,
                                                 shear_mass_profile=mass_profiles.ExternalShear)

    # Add the galaxy priors to the optimizer
    optimizer_1.source_galaxies = [source_galaxy_prior]
    optimizer_1.lens_galaxies = [lens_galaxy_prior]

    # Analyse the system with constant instrumentation
    result_1 = optimizer_1.fit(analysis, instrumentation=instrumentation)

    # Add the result of the first analysis to the list
    results.append(result_1)

    """
    2) Mass: SIE+Shear (priors from phase 1)
       Source: 'smooth' pixelization (include regularization parameter(s) in the model)
       NLO: LM
    """
    # Create an optimizer
    optimizer_2 = non_linear.DownhillSimplex()

    # Define galaxy priors
    lens_galaxy_prior = galaxy_prior.GalaxyPrior(spherical_mass_profile=mass_profiles.SphericalIsothermal,
                                                 shear_mass_profile=mass_profiles.ExternalShear)
    # The source galaxy is now represented by a pixelization
    source_galaxy_prior = galaxy_prior.GalaxyPrior(pixelization=pixelization.SquarePixelization)

    # Add the galaxy priors to the optimizer
    optimizer_2.lens_galaxies = [lens_galaxy_prior]
    optimizer_2.source_galaxies = [source_galaxy_prior]

    # Associate priors founds in the first analysis with the new galaxy priors
    lens_galaxy_prior_result = result_1.priors.lens_galaxies[0]
    lens_galaxy_prior.spherical_mass_profile = lens_galaxy_prior_result.spherical_mass_profile
    lens_galaxy_prior.shear_mass_profile = lens_galaxy_prior_result.shear_mass_profile

    # Analyse the system
    result_2 = optimizer_2.fit(analysis)

    # Add the result of the second analysis to the list
    results.append(result_2)

    """
    2H) Hyper-parameters: All included in model (most priors broad and uniform, but use previous phase regularization as well)
        Mass: SIE+Shear (Fixed to highest likelihood model from phase 2)
        Source: 'noisy' pixelization
        NLO: MN
    """
    # Create an optimizer
    optimizer_2h = non_linear.MultiNest()

    # Define a single galaxy prior that is a pixelized galaxy
    source_galaxy_prior = galaxy_prior.GalaxyPrior(pixelization=pixelization.VoronoiPixelization)

    #  Add the variable pixelization and instrumentation to the optimizer
    optimizer_2h.source_galaxies = [source_galaxy_prior]
    optimizer_2h.instrumentation = model_mapper.PriorModel(inst.Instrumentation)

    # Set the regularization prior using results from analysis 2
    source_galaxy_prior.pixelization.regularization = result_2.priors.pixelization.regularization

    # Analyse the system
    result_2h = optimizer_2h.fit(analysis, lens_galaxies=result_2.lens_galaxies)

    # Add the result of analysis 2h to the results
    results.append(result_2h)

    """
    a) Mass: SPLE+Shear (priors from Init phase 2)
       Source: 'noisy' pixelization (Fixed to init 2H hyper-parameters)
    """
    # Create an optimizer
    optimizer_a = non_linear.MultiNest()

    # Define a lens galaxy prior
    lens_galaxy_prior = galaxy_prior.GalaxyPrior(spherical_power_law_mass_profile=mass_profiles.SphericalPowerLaw,
                                                 shear_mass_profile=mass_profiles.ExternalShear)

    # Add the lens galaxy prior to the optimizer
    optimizer_a.lens_galaxies = [lens_galaxy_prior]

    # Set some lens galaxy priors using results from analysis 2
    lens_galaxy_prior_result = result_2.priors.lens_galaxies[0]
    lens_galaxy_prior.shear_mass_profile = lens_galaxy_prior_result.shear_mass_profile
    lens_galaxy_prior.spherical_power_law_mass_profile.centre = lens_galaxy_prior_result.spherical_mass_profile.centre

    # Analysis the system
    result_a = optimizer_a.fit(analysis, instrumentation=result_2h.instrumentation,
                               source_galaxies=result_2h.source_galaxies)

    # Add the result of the main analysis to the results
    results.append(result_a)

    # Return the results
    return results
