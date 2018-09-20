from autolens import exc
from autolens.lensing import galaxy
import inspect
from autolens.profiles import light_profiles, mass_profiles
from autolens.autofit import model_mapper


def is_light_profile_class(cls):
    """
    Parameters
    ----------
    cls
        Some object

    Returns
    -------
    bool: is_light_profile_class
        True iff cls is a class that inherits from light profile
    """
    return inspect.isclass(cls) and issubclass(cls, light_profiles.LightProfile)


def is_mass_profile_class(cls):
    """
    Parameters
    ----------
    cls
        Some object

    Returns
    -------
    bool: is_mass_profile_class
        True iff cls is a class that inherits from mass profile
    """
    return inspect.isclass(cls) and issubclass(cls, mass_profiles.MassProfile)


def is_profile_class(cls):
    """
    Parameters
    ----------
    cls
        Some object

    Returns
    -------
    bool: is_mass_profile_class
        True iff cls is a class that inherits from mass profile or light profile
    """
    return is_light_profile_class(cls) or is_mass_profile_class(cls)


class GalaxyPrior(model_mapper.AbstractPriorModel):
    """
    @DynamicAttrs
    """

    @property
    def flat_prior_models(self):
        return [flat_prior_model for prior_model in self.prior_models for flat_prior_model in
                prior_model.flat_prior_models]

    def __init__(self, align_centres=False, align_orientations=False, redshift=None, variable_redshift=False,
                 pixelization=None, regularization=None, hyper_galaxy=None, config=None, **kwargs):
        """
        Class to produce Galaxy instances from sets of profile classes using the model mapper

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

        if redshift is not None:
            self.redshift = model_mapper.Constant(
                redshift.redshift if isinstance(redshift, galaxy.Redshift) else redshift)
        else:
            self.redshift = model_mapper.PriorModel(galaxy.Redshift,
                                                    config) if variable_redshift else model_mapper.Constant(1)

        if pixelization is not None and regularization is None:
            raise exc.PriorException('If the galaxy prior has a pixelization, it must also have a regularization.')
        if pixelization is None and regularization is not None:
            raise exc.PriorException('If the galaxy prior has a regularization, it must also have a pixelization.')

        self.pixelization = model_mapper.PriorModel(pixelization, config) if inspect.isclass(
            pixelization) else pixelization
        self.regularization = model_mapper.PriorModel(regularization, config) if inspect.isclass(
            regularization) else regularization

        self.hyper_galaxy = model_mapper.PriorModel(hyper_galaxy, config) if inspect.isclass(
            hyper_galaxy) else hyper_galaxy
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
    def constant_light_profiles(self):
        """
        Returns
        -------
        light_profiles: [light_profiles.LightProfile]
            Light profiles with set variables
        """
        return [value for value in self.__dict__.values() if galaxy.is_light_profile(value)]

    @property
    def constant_mass_profiles(self):
        """
        Returns
        -------
        mass_profiles: [mass_profiles.MassProfile]
            Mass profiles with set variables
        """
        return [value for value in self.__dict__.values() if galaxy.is_mass_profile(value)]

    @property
    def prior_models(self):
        """
        Returns
        -------
        prior_models: [model_mapper.PriorModel]
            A list of the prior models (e.g. variable profiles) attached to this galaxy prior
        """
        return [value for _, value in
                filter(lambda t: isinstance(t[1], model_mapper.PriorModel), self.__dict__.items())]

    @property
    def profile_prior_model_dict(self):
        """
        Returns
        -------
        profile_prior_model_dict: {str: PriorModel}
            A dictionary mapping_matrix instance variable names to variable profiles.
        """
        return {key: value for key, value in
                filter(lambda t: isinstance(t[1], model_mapper.PriorModel) and is_profile_class(t[1].cls),
                       self.__dict__.items())}

    @property
    def constant_profile_dict(self):
        """
        Returns
        -------
        constant_profile_dict: {str: geometry_profiles.GeometryProfile}
            A dictionary mapping_matrix instance variable names to profiles with set variables.
        """
        return {key: value for key, value in self.__dict__.items() if
                galaxy.is_light_profile(value) or galaxy.is_mass_profile(value)}

    @property
    def light_profile_prior_model_dict(self):
        """
        Returns
        -------
        profile_prior_model_dict: {str: PriorModel}
            A dictionary mapping_matrix instance variable names to variable light profiles.
        """
        return {key: value for key, value in self.prior_model_dict.items() if is_light_profile_class(value.cls)}

    @property
    def mass_profile_prior_model_dict(self):
        """
        Returns
        -------
        profile_prior_model_dict: {str: PriorModel}
            A dictionary mapping_matrix instance variable names to variable mass profiles.
        """
        return {key: value for key, value in self.prior_model_dict.items() if is_mass_profile_class(value.cls)}

    @property
    def priors(self):
        """
        Returns
        -------
        priors: [Prior]
            A list of priors associated with prior models in this galaxy prior.
        """
        return [prior for prior_model in self.prior_models for prior in prior_model.priors]

    @property
    def prior_class_dict(self):
        """
        Returns
        -------
        prior_class_dict: {Prior: class}
            A dictionary mapping_matrix priors to the class associated with their prior model.
        """
        return {prior: cls for prior_model in self.prior_models for prior, cls in
                prior_model.prior_class_dict.items()}

    def instance_for_arguments(self, arguments):
        """
        Create an instance of the associated class for a set of arguments

        Parameters
        ----------
        arguments: {Prior: value}
            Dictionary mapping_matrix priors to attribute analysis_path and value pairs

        Returns
        -------
            An instance of the class
        """
        profiles = {**{key: value.instance_for_arguments(arguments)
                       for key, value
                       in self.profile_prior_model_dict.items()}, **self.constant_profile_dict}

        if isinstance(self.redshift, galaxy.Redshift):
            redshift = self.redshift
        else:
            redshift = self.redshift.instance_for_arguments(arguments)
        pixelization = self.pixelization.instance_for_arguments(arguments) \
            if isinstance(self.pixelization, model_mapper.PriorModel) \
            else self.pixelization
        regularization = self.regularization.instance_for_arguments(arguments) \
            if isinstance(self.regularization, model_mapper.PriorModel) \
            else self.regularization
        hyper_galaxy = self.hyper_galaxy.instance_for_arguments(arguments) \
            if isinstance(self.hyper_galaxy, model_mapper.PriorModel) \
            else self.hyper_galaxy

        return galaxy.Galaxy(redshift=redshift.redshift, pixelization=pixelization, regularization=regularization,
                             hyper_galaxy=hyper_galaxy, **profiles)

    def gaussian_prior_model_for_arguments(self, arguments):
        """
        Create a new galaxy prior from a set of arguments, replacing the priors of some of this galaxy prior's prior
        models with new arguments.

        Parameters
        ----------
        arguments: dict
            A dictionary mapping_matrix between old priors and their replacements.

        Returns
        -------
        new_model: GalaxyPrior
            A model with some or all priors replaced.
        """
        new_model = GalaxyPrior(align_centres=self.align_centres, align_orientations=self.align_orientations,
                                config=self.config)

        for key, value in filter(lambda t: isinstance(t[1], model_mapper.PriorModel), self.__dict__.items()):
            setattr(new_model, key, value.gaussian_prior_model_for_arguments(arguments))

        return new_model

    @classmethod
    def from_galaxy(cls, g, **kwargs):
        """
        Create a new galaxy prior with constants taken from a galaxy.

        Parameters
        ----------
        g: galaxy.Galaxy
            A galaxy
        kwargs
            Key word arguments to override GalaxyPrior constructor arguments.
        """
        return GalaxyPrior(**{**g.__dict__, **kwargs})
