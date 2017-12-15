class Prior(object):
    def __init__(self, name):
        self.name = name

    def argument_for(self, unit):
        return self.name, self.value_for(unit)

    def value_for(self, unit):
        raise AssertionError("Prior.value_for should be overridden")


class UniformPrior(Prior):
    def __init__(self, name, lower_limit=0., upper_limit=1.):
        super(UniformPrior, self).__init__(name)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def value_for(self, unit):
        return self.lower_limit + unit * (self.upper_limit - self.lower_limit)
