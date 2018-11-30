import inspect

from autofit import exc
from autofit.core import model_mapper
from autofit.core.model_mapper import PriorNameValue, ConstantNameValue, cast_collection

from autolens.model.galaxy import galaxy
from autolens.model.profiles import light_profiles, mass_profiles


def is_light_profile_class(cls):
    """
    Parameters
    ----------
    cls
        Some object

    Returns
    -------
    bool: is_light_profile_class
        True if cls is a class that inherits from light profile
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
        True if cls is a class that inherits from mass profile
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
        True if cls is a class that inherits from light profile or mass profile
    """
    return is_light_profile_class(cls) or is_mass_profile_class(cls)


class GalaxyModel(model_mapper.AbstractPriorModel):
    """
    @DynamicAttrs
    """

    @property
    def flat_prior_model_tuples(self):
        return [flat_prior_model for prior_model in self.prior_models for flat_prior_model in
                prior_model.flat_prior_model_tuples]

    def __init__(self, align_centres=False, align_axis_ratios=False, align_orientations=False, redshift=None,
                 variable_redshift=False, pixelization=None, regularization=None, hyper_galaxy=None, **kwargs):
        """Class to produce Galaxy instances from sets of profile classes and other model-fitting attributes (e.g. \
         pixelizations, regularization schemes, hyper-galaxyes) using the model mapper.

        Parameters
        ----------
        light_profile_classes: [LightProfile]
            The *LightProfile* classes for which model light profile instances are generated for this galaxy model.
        mass_profile_classes: [MassProfile]
            The *MassProfile* classes for which model mass profile instances are generated for this galaxy model.
        align_centres : bool
            If *True*, the same prior will be used for all the profiles centres, such that all light and / or mass \
            profiles always share the same origin.
        align_axis_ratios : bool
            If *True*, the same prior will be used for all the profiles axis-ratio, such that all light and / or mass \
            profiles always share the same axis-ratio.
        align_orientations : bool
            If *True*, the same prior will be used for all the profiles rotation angles phi, such that all light \
            and / or mass profiles always share the same orientation.
        redshift : float
            The redshift of this model galaxy.
        variable_redshift : bool
            If *True*, the galaxy redshift will be treated as a free-parameter that is fitted for by the non-linear \
            search.
        pixelization : Pixelization
            The pixelization used to reconstruct the galaxy light and fit the observed regular if using an inversion.
        regularization : Regularization
            The regularization-scheme used to regularization reconstruct the galaxy light when fitting the observed \
            regular if using an inversion.
        hyper_galaxy : HyperGalaxy
            A model hyper-galaxy used for scaling the observed regular's noise.
        """

        self.align_centres = align_centres
        self.align_axis_ratios = align_axis_ratios
        self.align_orientations = align_orientations

        profile_models = []

        for name, cls in kwargs.items():
            if is_mass_profile_class(cls) or is_light_profile_class(cls):
                model = model_mapper.PriorModel(cls)
                profile_models.append(model)
                setattr(self, name, model)
            else:
                setattr(self, name, cls)

        if len(profile_models) > 0:
            if self.align_centres:
                centre = profile_models[0].centre
                for profile_model in profile_models:
                    profile_model.centre = centre

            if self.align_axis_ratios:
                axis_ratio = profile_models[0].axis_ratio
                for profile_model in profile_models:
                    profile_model.axis_ratio = axis_ratio

            if self.align_orientations:
                phi = profile_models[0].phi
                for profile_model in profile_models:
                    profile_model.phi = phi

        if redshift is not None:
            self.redshift = model_mapper.Constant(
                redshift.redshift if isinstance(redshift, galaxy.Redshift) else redshift)
        else:
            self.redshift = model_mapper.PriorModel(galaxy.Redshift) if variable_redshift else model_mapper.Constant(1)

        if pixelization is not None and regularization is None:
            raise exc.PriorException('If the galaxy prior has a pixelization, it must also have a regularization.')
        if pixelization is None and regularization is not None:
            raise exc.PriorException('If the galaxy prior has a regularization, it must also have a pixelization.')

        self.pixelization = model_mapper.PriorModel(pixelization) if inspect.isclass(
            pixelization) else pixelization
        self.regularization = model_mapper.PriorModel(regularization) if inspect.isclass(
            regularization) else regularization

        self.hyper_galaxy = model_mapper.PriorModel(hyper_galaxy) if inspect.isclass(hyper_galaxy) else hyper_galaxy

    def __setattr__(self, key, value):
        if key == "redshift" \
                and (isinstance(value, float)
                     or isinstance(value, int)
                     or isinstance(value, model_mapper.Prior)
                     or isinstance(value, model_mapper.Constant)):
            value = galaxy.Redshift(value)
        super(GalaxyModel, self).__setattr__(key, value)

    def linked_model_for_classes(self, **classes):
        light_profile_class_tuples = [(name, cls) for name, cls in classes.items() if is_light_profile_class(cls)]
        mass_profile_class_tuples = [(name, cls) for name, cls in classes.items() if is_mass_profile_class(cls)]

        def link_models_to_classes(models, class_tuples):
            result = []
            for n, class_tuple in enumerate(class_tuples):
                result.append((class_tuple[0], models[n % len(models)].linked_model_for_class(class_tuple[1])))
            return result

        light_profile_model_tuples = link_models_to_classes(self.light_profile_prior_models, light_profile_class_tuples)
        mass_profile_model_tuples = link_models_to_classes(self.mass_profile_prior_models, mass_profile_class_tuples)

        return GalaxyModel(**{**dict(light_profile_model_tuples), **dict(mass_profile_model_tuples)})

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
    def light_profile_prior_models(self):
        return [item for item in self.__dict__.values() if
                isinstance(item, model_mapper.PriorModel) and is_light_profile_class(item.cls)]

    @property
    def mass_profile_prior_models(self):
        return [item for item in self.__dict__.values() if
                isinstance(item, model_mapper.PriorModel) and is_mass_profile_class(item.cls)]

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
    @cast_collection(PriorNameValue)
    def prior_tuples(self):
        """
        Returns
        -------
        priors: [PriorTuple]
            A list of priors associated with prior models in this galaxy prior.
        """
        return [prior for prior_model in self.prior_models for prior in prior_model.prior_tuples]

    @property
    @cast_collection(ConstantNameValue)
    def constant_tuples(self):
        """
        Returns
        -------
        constant: [ConstantTuple]
            A list of constants associated with prior models in this galaxy prior.
        """
        return [constant for prior_model in self.prior_models for constant in prior_model.constant_tuples]

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
        new_model: GalaxyModel
            A model with some or all priors replaced.
        """
        new_model = GalaxyModel(align_centres=self.align_centres, align_axis_ratios=self.align_axis_ratios,
                                align_orientations=self.align_orientations)

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
            Key word arguments to override GalaxyModel constructor arguments.
        """
        return GalaxyModel(**{**g.__dict__, **kwargs})
