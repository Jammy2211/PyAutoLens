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


class ClassMappingPriorCollection(PriorCollection):
    def __init__(self, config):
        super(ClassMappingPriorCollection, self).__init__()
        self.class_priors = []
        self.classes = []
        self.config = config

    def add_class(self, cls, *priors):
        args = inspect.getargspec(cls.__init__).args[1:]

        priors_for_class = []
        for arg in args:
            matching_priors = filter(lambda p: p.name == arg, priors)
            if len(matching_priors) > 0:
                prior = matching_priors[0]
            else:
                config_arr = self.config.get(cls.__name__, arg)
                prior = UniformPrior("{}.{}".format(len(self.classes), arg), config_arr[1], config_arr[2])
            priors_for_class.append(prior)
            self.add(prior)

        self.classes.append(cls)
        self.class_priors.append(priors_for_class)

        return priors_for_class
