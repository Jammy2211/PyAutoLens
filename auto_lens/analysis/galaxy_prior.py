from auto_lens.analysis import galaxy
from auto_lens import exc
from auto_lens.profiles import light_profiles, mass_profiles
from auto_lens.analysis import model_mapper


def is_light_profile(cls):
    return issubclass(cls, light_profiles.LightProfile) and not issubclass(cls, mass_profiles.MassProfile)


def is_mass_profile(cls):
    return issubclass(cls, mass_profiles.MassProfile)


class GalaxyPrior(model_mapper.AbstractPriorModel):
    """
    Class to produce Galaxy instances from sets of profile classes using the model mapper
    """

    def __init__(self, align_centres=False, align_orientations=False, config=None, **kwargs):
        """
        Parameters
        ----------
        light_profile_classes: [LightProfile]
            The classes for which light profile instances are generated for this galaxy
        mass_profile_classes: [MassProfile]
            The classes for which light profile instances are generated for this galaxy
        align_centres: Bool
            If True the same prior will be used for all the profiles centres such that any generated profiles always
            have the same centre
        align_orientations: Bool
            If True the same prior will be used for all the profiles orientations such that any generated profiles
            always have the same orientation
        """

        self.light_profile_dict = {key: value for key, value in kwargs.items() if
                                   is_light_profile(value)}
        self.mass_profile_dict = {key: value for key, value in kwargs.items() if
                                  is_mass_profile(value)}

        self.align_centres = align_centres
        self.align_orientations = align_orientations

        profile_models = []

        for name, cls in kwargs.items():
            model = model_mapper.PriorModel(cls, config)
            profile_models.append(model)
            setattr(self, name, model)

        if self.align_centres:
            centre = profile_models[0].centre
            for profile_model in profile_models:
                profile_model.centre = centre

        if self.align_orientations:
            phi = profile_models[0].phi
            for profile_model in profile_models:
                profile_model.phi = phi

        self.redshift = model_mapper.PriorModel(galaxy.Redshift, config)

    @property
    def light_profile_names(self):
        return list(self.light_profile_dict.keys())

    @property
    def mass_profile_names(self):
        return list(self.mass_profile_dict.keys())

    @property
    def prior_models(self):
        return [value for _, value in
                filter(lambda t: isinstance(t[1], model_mapper.PriorModel), self.__dict__.items())]

    @property
    def light_profile_prior_models(self):
        return filter(
            lambda prior_model: is_light_profile(prior_model.cls), self.prior_models)

    @property
    def mass_profile_prior_models(self):
        return filter(
            lambda prior_model: is_mass_profile(prior_model.cls), self.prior_models)

    @property
    def priors(self):
        return [prior for prior_model in self.prior_models for prior in prior_model.priors]

    def instance_for_arguments(self, arguments):
        """
        Create an instance of the associated class for a set of arguments

        Parameters
        ----------
        arguments: {Prior: value}
            Dictionary mapping priors to attribute name and value pairs

        Returns
        -------
            An instance of the class
        """
        instance_light_profiles = list(map(lambda prior_model: prior_model.instance_for_arguments(arguments),
                                           self.light_profile_prior_models))
        instance_mass_profiles = list(map(lambda prior_model: prior_model.instance_for_arguments(arguments),
                                          self.mass_profile_prior_models))
        instance_redshift = self.redshift.instance_for_arguments(arguments)
        return galaxy.Galaxy(light_profiles=instance_light_profiles, mass_profiles=instance_mass_profiles,
                             redshift=instance_redshift.redshift)

    # def gaussian_prior_model_for_arguments(self, prior_arguments):
    #     new_model = PriorModel(self.cls, self.config)
    #
    #     for tuple_prior in self.tuple_priors:
    #         setattr(new_model, tuple_prior[0], tuple_prior[1].gaussian_tuple_prior_for_arguments(prior_arguments))
    #     for prior in self.direct_priors:
    #         setattr(new_model, prior[0], prior_arguments[prior[0]])
    #
    #     return new_model
