class UniformPrior(object):
    def __init__(self, name, lower_limit=0., upper_limit=1.):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def value_for(self, unit):
        return self.lower_limit + unit * (self.upper_limit - self.lower_limit)
