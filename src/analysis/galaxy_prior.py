from src.analysis import galaxy
import inspect
from src.profiles import light_profiles, mass_profiles
from src.analysis import model_mapper


def is_light_profile_class(cls):
    return inspect.isclass(cls) and issubclass(
        cls, light_profiles.LightProfile) and not issubclass(
        cls, mass_profiles.MassProfile)


def is_mass_profile_class(cls):
    return inspect.isclass(cls) and issubclass(cls, mass_profiles.MassProfile)


def is_profile_class(cls):
    return inspect.isclass(cls) \
           and issubclass(cls, mass_profiles.MassProfile) \
           or issubclass(cls, light_profiles.LightProfile)


# TODO : Add in init a 'vary redshift' bool which is default False

class GalaxyPrior(model_mapper.AbstractPriorModel):
    """
    Class to produce Galaxy instances from sets of profile classes using the model mapper
    @DynamicAttrs
    """

    @property
    def flat_prior_models(self):
        return [flat_prior_model for prior_model in self.prior_models for flat_prior_model in
                prior_model.flat_prior_models]

    def __init__(self, align_centres=False, align_orientations=False, pixelization=None, hyper_galaxy=None, config=None,
                 **kwargs):
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

        self.align_centres = align_centres
        self.align_orientations = align_orientations

        profile_models = []

        for name, cls in kwargs.items():
            if is_mass_profile_class(cls) or is_light_profile_class(cls):
                model = model_mapper.PriorModel(cls, config)
                profile_models.append(model)
                setattr(self, name, model)
            else:
                setattr(self, name, cls)

        if len(profile_models) > 0:
            if self.align_centres:
                centre = profile_models[0].centre
                for profile_model in profile_models:
                    profile_model.centre = centre

            if self.align_orientations:
                phi = profile_models[0].phi
                for profile_model in profile_models:
                    profile_model.phi = phi

        self.redshift = model_mapper.PriorModel(galaxy.Redshift, config)
        self.pixelization = model_mapper.PriorModel(pixelization, config) if pixelization is not None else None
        self.hyper_galaxy = model_mapper.PriorModel(hyper_galaxy, config) if hyper_galaxy is not None else None
        self.config = config

    def __setattr__(self, key, value):
        if key == "redshift" \
                and (isinstance(value, float)
                     or isinstance(value, int)
                     or isinstance(value, model_mapper.Prior)
                     or isinstance(value, model_mapper.Constant)):
            value = galaxy.Redshift(value)
        super(GalaxyPrior, self).__setattr__(key, value)

    @property
    def fixed_light_profiles(self):
        return [value for value in self.__dict__.values() if galaxy.is_light_profile(value)]

    @property
    def fixed_mass_profiles(self):
        return [value for value in self.__dict__.values() if galaxy.is_mass_profile(value)]

    @property
    def prior_models(self):
        return [value for _, value in
                filter(lambda t: isinstance(t[1], model_mapper.PriorModel), self.__dict__.items())]

    @property
    def profile_prior_model_dict(self):
        return {key: value for key, value in
                filter(lambda t: isinstance(t[1], model_mapper.PriorModel) and is_profile_class(t[1].cls),
                       self.__dict__.items())}

    @property
    def fixed_profile_dict(self):
        return {key: value for key, value in self.__dict__.items() if
                galaxy.is_light_profile(value) or galaxy.is_mass_profile(value)}

    @property
    def light_profile_prior_model_dict(self):
        return {key: value for key, value in self.prior_model_dict.items() if is_light_profile_class(value.cls)}

    @property
    def mass_profile_prior_model_dict(self):
        return {key: value for key, value in self.prior_model_dict.items() if is_mass_profile_class(value.cls)}

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
        profiles = {**{key: value.instance_for_arguments(arguments)
                       for key, value
                       in self.profile_prior_model_dict.items()}, **self.fixed_profile_dict}

        instance_redshift = self.redshift.instance_for_arguments(arguments)
        pixelization = self.pixelization.instance_for_arguments(arguments) if self.pixelization is not None else None
        hyper_galaxy = self.hyper_galaxy.instance_for_arguments(arguments) if self.hyper_galaxy is not None else None

        return galaxy.Galaxy(redshift=instance_redshift.redshift, pixelization=pixelization, hyper_galaxy=hyper_galaxy,
                             **profiles)

    def gaussian_prior_model_for_arguments(self, arguments):
        new_model = GalaxyPrior(align_centres=self.align_centres, align_orientations=self.align_orientations,
                                config=self.config)

        for key, value in filter(lambda t: isinstance(t[1], model_mapper.PriorModel), self.__dict__.items()):
            setattr(new_model, key, value.gaussian_prior_model_for_arguments(arguments))

        return new_model
