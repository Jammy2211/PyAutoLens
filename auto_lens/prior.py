class Prior(object):
    def __init__(self, name):
        self.name = name

    def argument_for(self, unit):
        return self.name, self.value_for(unit)

    def value_for(self, unit):
        raise AssertionError("Prior.value_for should be overridden")

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return self.name != other.name

    def __hash__(self):
        return hash(self.name)


class UniformPrior(Prior):
    def __init__(self, name, lower_limit=0., upper_limit=1.):
        super(UniformPrior, self).__init__(name)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def value_for(self, unit):
        return self.lower_limit + unit * (self.upper_limit - self.lower_limit)


class PriorCollection(list):
    def __init__(self, *priors):
        super(PriorCollection, self).__init__(priors)

    def arguments_for_vector(self, vector):
        return dict(map(lambda prior, unit: prior.argument_for(unit), self, vector))

    def append(self, p_object):
        if p_object in self:
            self[self.index(p_object)] = p_object
        else:
            super(PriorCollection, self).append(p_object)
