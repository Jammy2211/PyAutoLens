from auto_lens.analysis import model_mapper as mm
from auto_lens.analysis import galaxy_prior as gp
from auto_lens.profiles import light_profiles, mass_profiles
import os


class TestCase:
    def test_integration(self):
        config = mm.Config("{}/../{}".format(os.path.dirname(os.path.realpath(__file__)), "test_files/config"))

        # Create a mapper. This can be used to convert values output by a non linear optimiser into class instances.
        mapper = mm.ModelMapper()

        # Create a galaxy prior for the source galaxy. Here we are describing only the light profile of the source
        # galaxy which comprises an elliptical exponential and elliptical sersic light profile.
        source_galaxy_prior = gp.GalaxyPrior(
            light_profile_one=light_profiles.EllipticalExponential,
            light_profile_2=light_profiles.EllipticalSersic, config=config)

        # Create a galaxy prior for the source galaxy. Here we are describing both the light and mass profiles. We've
        # also stipulated that the centres of any galaxies generated using the galaxy prior should match.
        lens_galaxy_prior = gp.GalaxyPrior(light_profile=light_profiles.EllipticalExponential,
                                           mass_profile=mass_profiles.EllipticalExponentialMass,
                                           align_centres=True, config=config)

        mapper.source_galaxy = source_galaxy_prior
        mapper.lens_galaxy = lens_galaxy_prior

        # Create a model instance. All the instances of the profile classes are created here. Normally we would do this
        # using the output of a non linear search but in this case we are using the median values from the priors.
        instance = mapper.physical_values_from_prior_medians()

        # Recover galaxy instances. We can pass the model instance to galaxy priors to recover a fully constructed
        # galaxy
        source_galaxy = source_galaxy_prior.galaxy_for_model_instance(instance)
        lens_galaxy = lens_galaxy_prior.galaxy_for_model_instance(instance)

        # Let's just check that worked
        assert len(source_galaxy.light_profiles) == 2
        assert len(source_galaxy.mass_profiles) == 0

        assert len(lens_galaxy.light_profiles) == 1
        assert len(lens_galaxy.mass_profiles) == 1

        assert source_galaxy.redshift == 1.5
        assert lens_galaxy.redshift == 1.5
