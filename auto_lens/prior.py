import math
from scipy.special import erfinv
import inspect


class Prior(object):
    """Defines a prior that converts unit hypercube values into argument values"""

    def __init__(self, path):
        """

        Parameters
        ----------
        path: String
            The name of the attribute to which this prior is associated
        """
        self.path = path

    def argument_for(self, unit):
        """

        Parameters
        ----------
        unit: Int
            A unit hypercube value between 0 and 1
        Returns
        -------
        argument: (String, float)
            Returns the name of an attribute and its calculated value as a tuple
        """
        return self.name, self.value_for(unit)

    @property
    def name(self):
        return self.path.split(".")[-1]

    def value_for(self, unit):
        raise AssertionError("Prior.value_for should be overridden")

    def __eq__(self, other):
        return self.path == other.path

    def __ne__(self, other):
        return self.path != other.path

    def __hash__(self):
        return hash(self.path)

    def __repr__(self):
        return "<Prior path={}>".format(self.path)


class UniformPrior(Prior):
    """A prior with a uniform distribution between a lower and upper limit"""

    def __init__(self, path, lower_limit=0., upper_limit=1.):
        """

        Parameters
        ----------
        path: String
            The attribute name
        lower_limit: Float
            The lowest value this prior can return
        upper_limit: Float
            The highest value this prior can return
        """
        super(UniformPrior, self).__init__(path)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def value_for(self, unit):
        """

        Parameters
        ----------
        unit: Float
            A unit hypercube value between 0 and 1
        Returns
        -------
        value: Float
            A value for the attribute between the upper and lower limits
        """
        return self.lower_limit + unit * (self.upper_limit - self.lower_limit)


class GaussianPrior(Prior):
    """A prior with a gaussian distribution"""

    def __init__(self, path, mean, sigma):
        super(GaussianPrior, self).__init__(path)
        self.mean = mean
        self.sigma = sigma

    def value_for(self, unit):
        """

        Parameters
        ----------
        unit: Float
            A unit hypercube value between 0 and 1
        Returns
        -------
        value: Float
            A value for the attribute biased to the gaussian distribution
        """
        return self.mean + (self.sigma * math.sqrt(2) * erfinv((unit * 2.0) - 1.0))


class PriorCollection(list):
    """A collection of priors, perhaps associated with one component of the model (e.g. lens mass distribution)"""

    def __init__(self, *priors):
        """

        Parameters
        ----------
        priors: [Prior]
            A list of priors
        """
        super(PriorCollection, self).__init__(priors)

    def arguments_for_vector(self, vector):
        """
        Used to obtain a dictionary of attribute values that can then be passed to construct a class.

        Examples
        --------
        # This constructs a new profile from a collection of priors. Note the prior collection must contain a prior for
        # each argument of the function the arguments are passed into.
        
        p = profile.Profile(**collection.arguments_for_vector(vector))

        Parameters
        ----------
        vector: [Float]
            A vector of hypercube unit values. Note that this must match the order of associated priors in the
            collection.

        Returns
        -------
        arguments: {String: Float}
            A dictionary of attribute names and associated values
        """
        if len(vector) != len(self):
            raise AssertionError("PriorCollection and unit vector have different lengths")
        return dict(map(lambda prior, unit: prior.argument_for(unit), self, vector))

    def add(self, prior):
        """
        Add a prior to the collection. If a prior with the same name is already in the collection then this prior will
        replace it with the same index.

        Parameters
        ----------
        prior: Prior
            A prior to add to this collection

        """
        if prior in self:
            self[self.index(prior)] = prior
        else:
            super(PriorCollection, self).append(prior)

    def append(self, p_object):
        raise AssertionError("Append should not be called directly")


class PriorModel(object):
    """Object comprising class, name and associated priors"""

    def __init__(self, name, cls):
        """
        Parameters
        ----------
        name: String
            The name of this prior model instance (e.g. sersic_profile_1)
        cls: class
            The class associated with this instance
        """
        self.name = name
        self.cls = cls

    @property
    def priors(self):
        return filter(lambda v: isinstance(v, Prior), self.__dict__.values())

    def instance_for_arguments(self, arguments):
        """
        Create an instance of the associated class for a set of arguments

        Parameters
        ----------
        arguments: {Prior: (str: any)}
            Dictionary mapping priors to attribute name and value pairs

        Returns
        -------
            An instance of the class
        """
        model_arguments = {arguments[val][0]: arguments[val][1] for val in self.__dict__.values() if val in arguments}
        return self.cls(**model_arguments)


class Reconstruction(object):
    pass


# TODO: Test config loading and implement inherited attribute setting. Priors that can produce Tuples?
class ClassMappingPriorCollection(object):
    """A collection of priors formed by passing in classes to be reconstructed"""

    def __init__(self, config):
        """
        Parameters
        ----------
        config: Config
            An object that wraps a configuration

        Examples
        --------
        # A ClassMappingPriorCollection keeps track of priors associated with the attributes required to construct
        # instances of classes.

        # A config is passed into the collection to provide default setup values for the priors:

        collection = ClassMappingPriorCollection(config)

        # All class instances that are to be generated by the collection are specified by passing their name and class:

        collection.add_class("sersic_1", light_profile.SersicLightProfile)
        collection.add_class("sersic_2", light_profile.SersicLightProfile)
        collection.add_class("other_instance", SomeClass)

        # A PriorModel instance is created each time we add a class to the collection. We can access those models using
        # their name:

        sersic_model_1 = collection.sersic_1

        # This allows us to replace the default priors:

        collection.sersic_1.intensity = GaussianPrior("Intensity", 2., 5.)

        # Or maybe we want to tie two priors together:

        collection.sersic_1.intensity = collection.sersic_2.intensity

        # This statement reduces the number of priors by one and means that the two sersic instances will always share
        # the same centre.

        # We can then create instances of every class for a unit hypercube vector with length equal to
        # len(collection.priors):

        reconstruction = collection.reconstruction_for_vector([.4, .2, .3, .1])

        # The attributes of the reconstruction are named the same as those of the collection:

        sersic_1 = collection.sersic_1

        # But this attribute is an instance of the actual SersicLightProfile class
        """
        super(ClassMappingPriorCollection, self).__init__()
        self.prior_models = []
        self.config = config

    def add_class(self, name, cls):
        """
        Add a class to this collection. Priors are automatically generated for __init__ arguments. Prior type and
        configuration is taken from matching module.class.attribute entries in the config.

        Parameters
        ----------
        name: String
            The name of this class. This is also the attribute name for the class in the collection and reconstruction.
        cls: class
            The class for which priors are to be generated.

        """

        args = inspect.getargspec(cls.__init__).args[1:]

        prior_model = PriorModel(name, cls)

        priors_for_class = []

        def add_prior(prior_name):
            config_arr = self.config.get(cls.__name__, prior_name)
            path = "{}.{}".format(len(self.prior_models), prior_name)
            if config_arr[0] == "u":
                prior = UniformPrior(path, config_arr[1], config_arr[2])
            elif config_arr[0] == "g":
                prior = GaussianPrior(path, config_arr[1], config_arr[2])

            priors_for_class.append(prior)
            setattr(prior_model, prior_name, prior)

        for arg in args:
            print(arg)
            if arg == "centre":
                add_prior("centre_x")
                add_prior("centre_y")
            else:
                add_prior(arg)

        setattr(self, name, prior_model)

        self.prior_models.append(prior_model)

    @property
    def prior_set(self):
        """
        Returns
        -------
        prior_set: set()
            The set of all priors associated with this collection
        """
        return {prior for prior_model in self.prior_models for prior in prior_model.priors}

    @property
    def priors(self):
        """
        Returns
        -------
        priors: [Prior]
            An ordered list of unique priors associated with this collection
        """
        return sorted(list(self.prior_set), key=lambda prior: prior.path)

    def reconstruction_for_vector(self, vector):
        """
        Creates a Reconstruction, which has an attribute and class instance corresponding to every PriorModel attributed
        to this instance.

        Parameters
        ----------
        vector: [float]
            A unit hypercube vector

        Returns
        -------
        reconstruction: Reconstruction
            An object containing reconstructed model instances

        """
        arguments = dict(map(lambda prior, unit: (prior, prior.argument_for(unit)), self.priors, vector))

        reconstruction = Reconstruction()

        for prior_model in self.prior_models:
            prior_model.instance_for_arguments(arguments)
            setattr(reconstruction, prior_model.name, prior_model.instance_for_arguments(arguments))

        return reconstruction
