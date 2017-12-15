import prior


class TestUniformPrior(object):
    def test__simple_assumptions(self):
        uniform_prior = prior.UniformPrior("test", lower_limit=0., upper_limit=1.)

        assert uniform_prior.value_for(0.) == 0.
        assert uniform_prior.value_for(1.) == 1.
        assert uniform_prior.value_for(0.5) == 0.5

    def test__non_zero_lower_limit(self):
        uniform_prior = prior.UniformPrior("test", lower_limit=0.5, upper_limit=1.)

        assert uniform_prior.value_for(0.) == 0.5
        assert uniform_prior.value_for(1.) == 1.
        assert uniform_prior.value_for(0.5) == 0.75


class TestArguments(object):
    def test__argument(self):
        uniform_prior = prior.UniformPrior("test", lower_limit=0., upper_limit=1.)

        assert uniform_prior.argument_for(0.) == ("test", 0)
        assert uniform_prior.argument_for(0.5) == ("test", 0.5)
