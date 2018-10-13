import os

from autolens import conf
from autolens.autofit import model_mapper as mm
from autolens.lensing import galaxy_model as gp
from autolens.profiles import light_profiles, mass_profiles


class TestCase:
    def test_integration(self):
        config = conf.DefaultPriorConfig(
            "{}/../{}".format(os.path.dirname(os.path.realpath(__file__)), "test_files/configs/galaxy_model/priors/default"))

        # Create a mapper. This can be used to convert values output by a non linear optimiser into class instances.
        mapper = mm.ModelMapper()

        # Create a galaxy prior for the source galaxy. Here we are describing only the light profile of the source
        # galaxy which comprises an elliptical exponential and elliptical sersic light profile.
        source_galaxy_prior = gp.GalaxyModel(variable_redshift=True,
                                             light_profile_one=light_profiles.EllipticalExponential,
                                             light_profile_2=light_profiles.EllipticalSersic, config=config)

        # Create a galaxy prior for the source galaxy. Here we are describing both the light and mass profiles. We've
        # also stipulated that the centres of any galaxies generated using the galaxy prior should match.
        lens_galaxy_prior = gp.GalaxyModel(variable_redshift=True, light_profile=light_profiles.EllipticalExponential,
                                           mass_profile=mass_profiles.EllipticalExponential,
                                           align_centres=True, config=config)

        mapper.source_galaxy = source_galaxy_prior
        mapper.lens_galaxy = lens_galaxy_prior

        # Create a model instance. All the instances of the profile classes are created here. Normally we would do this
        # using the output of a non linear search but in this case we are using the median values from the priors.
        instance = mapper.instance_from_prior_medians()

        # Recover galaxy instances. We can pass the model instance to galaxy priors to recover a fully constructed
        # galaxy
        source_galaxy = instance.source_galaxy
        lens_galaxy = instance.lens_galaxy

        # Let's just check that worked
        assert len(source_galaxy.light_profiles) == 2
        assert len(source_galaxy.mass_profiles) == 0

        assert len(lens_galaxy.light_profiles) == 1
        assert len(lens_galaxy.mass_profiles) == 1

        assert source_galaxy.redshift == 1.5
        assert lens_galaxy.redshift == 1.5
