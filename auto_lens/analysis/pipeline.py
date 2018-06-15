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
    mapper_1 = mm.ModelMapper()
    optimizer_1 = non_linear.LevenbergMarquardt(mapper_1)

    source_galaxy_prior = galaxy_prior.GalaxyPrior("source_galaxy_prior", mapper_1,
                                                   light_profile=light_profiles.EllipticalSersic)
    lens_galaxy_prior = galaxy_prior.GalaxyPrior("lens_galaxy_prior", mapper_1,
                                                 spherical_mass_profile=mass_profiles.SphericalIsothermal,
                                                 shear_mass_profile=mass_profiles.ExternalShear)

    initialization_1 = analysis.Analysis(mapper_1, non_linear_optimizer=optimizer_1,
                                         source_galaxy_priors=[source_galaxy_prior],
                                         lens_galaxy_priors=[lens_galaxy_prior])

    result_1 = initialization_1.run(image, mask, instrumentation=instrumentation)
    prior_result_1 = mapper_1.prior_results_for_gaussian_tuples(optimizer_1.compute_gaussian_priors(SIGMA_LIMIT))

    results.append(result_1)

    """
    2) Mass: SIE+Shear (priors from phase 1)
       Source: 'smooth' pixelization (include regularization parameter(s) in the model)
       NLO: LM
    """
    mapper_2 = mm.ModelMapper()
    optimizer_2 = non_linear.LevenbergMarquardt(mapper_2)

    lens_galaxy_prior = galaxy_prior.GalaxyPrior("lens_galaxy_prior", mapper_2,
                                                 spherical_mass_profile=mass_profiles.SphericalIsothermal,
                                                 shear_mass_profile=mass_profiles.ExternalShear)

    lens_galaxy_prior.override_prior_models(spherical_mass_profile=prior_result_1.spherical_mass_profile,
                                            shear_mass_profile=prior_result_1.shear_mass_profile)

    initialization_2 = analysis.Analysis(mapper_2, non_linear_optimizer=optimizer_2,
                                         lens_galaxy_priors=[lens_galaxy_prior],
                                         pixelization_class=pixelization.SquarePixelization)

    result_2 = initialization_2.run(image, mask, instrumentation=instrumentation)
    prior_result_2 = mapper_1.prior_results_for_gaussian_tuples(optimizer_1.compute_gaussian_priors(SIGMA_LIMIT))
    results.append(result_2)

    """
    2H) Hyper-parameters: All included in model (most priors broad and uniform, but use previous phase regularization as well)
        Mass: SIE+Shear (Fixed to highest likelihood model from phase 2)
        Source: 'noisy' pixelization
        NLO: MN
    """
    mapper_2h = mm.ModelMapper()
    optimizer_2h = non_linear.MultiNest(mapper_2h)

    initialization_2h = analysis.Analysis(mapper_2h, non_linear_optimizer=optimizer_2h,
                                          pixelization_class=pixelization.VoronoiPixelization,
                                          instrumentation_class=inst.Instrumentation)

    mapper_2h.pixelization.regularization = prior_result_2.pixelization.regularization

    result_2h = initialization_2h.run(image, mask, lens_galaxies=results.lens_galaxies,
                                      source_galaxies=results.source_galaxies)
    # prior_result_2h = mapper_1.prior_results_for_gaussian_tuples(optimizer_1.compute_gaussian_priors(SIGMA_LIMIT))
    results.append(result_2h)

    """
    a) Mass: SPLE+Shear (priors from Init phase 2)
       Source: 'noisy' pixelization (Fixed to init 2H hyper-parameters)
    """
    mapper_a = mm.ModelMapper()
    optimizer_a = non_linear.MultiNest(mapper_a)

    lens_galaxy_prior = galaxy_prior.GalaxyPrior("lens_galaxy_prior", mapper_a,
                                                 spherical_power_law_mass_profile=mass_profiles.SphericalPowerLaw,
                                                 shear_mass_profile=mass_profiles.ExternalShear)

    lens_galaxy_prior.override_prior_models(shear_mass_profile=prior_result_2.shear_mass_profile)
    lens_galaxy_prior.spherical_power_law_mass_profile.centre = prior_result_2.spherical_mass_profile.centre

    main_analysis = analysis.Analysis(mapper_a, non_linear_optimizer=optimizer_a,
                                      lens_galaxy_priors=[lens_galaxy_prior])
    result_a = main_analysis.run(image, mask, instrumentation=result_2h.instrumentation,
                                 pixelization=result_2h.pixelization)
    results.append(result_a)

    return results
