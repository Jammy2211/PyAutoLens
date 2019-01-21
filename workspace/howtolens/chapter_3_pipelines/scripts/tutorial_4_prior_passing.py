# In the previous 3  pipelines, we passed priors using the 'variable' attribute of the previous results. However, its
# not yet clear how these priors are passed. Do they use a UniformPrior or GaussianPrior? What are the limits / mean /
# width of these priors? Can I change this behaviour?

# Lets say I link two parameters in pass priors (don't run this code its just a demo)

def pass_priors(self, previous_results):

    self.lens_galaxies.galaxy_name.profile_name.parameter_name = \
        previous_results[0].variable.galaxy_name.profile_name.parameter_name

# By invoking the 'variable' attribute, the passing of priors behaves following 3 rules:

# 1) The 'self.lens_galaxies.galaxy_name.profile_name.parameter_name' parameter will use a GaussianPrior as its prior.

#    A GaussianPrior is ideal, as the 1D pdf results we compute at the end of a phase are easily summarized as a
#    Gaussian.

# 2) The mean of the GaussianPrior is the best-fit value of
#    'previous_results[0].variable.galaxy_name.profile_name.parameter_name'.

#    This means that MultiNest specifically starts by searching the region of non-linear parameter space that
#    corresponded to highest likelihood solutions in the previous phase. Thus, we're setting our priors to look in the
#    correct regions of parameter space.

# 3) The sigma of the Gaussian will use either: (i) the 1D error on the previous result's parameter or; (ii) the value
#    specified in the appropriate 'config/priors/width/profile.ini' config file (check these files out now).

#    The idea here is simple. We want a value of sigma that gives a GaussianPrior wide enough to search a broad region
#    of parameter space, so that the lens model can change if a better solution is nearby. However, we want it to be
#    narrow enough that we don't search too much of parameter space, as this will be slow or risk leading us into an
#    incorrect solution! A natural choice is the errors of the parameter from the previous phase.

#    Unfortunately, this doesn't always work. Lens modeling is prone to an effect called 'over-fitting' where we
#    underestimate the errors on our lens model parameters. This is especially true when we take the shortcuts we're
#    used to in early phases - aggresive masking, reduced data, simplified lens models, constant efficiency mode, etc.

#    Therefore, the priors/widths file is our fallback. If the error on a parameter is suspiciously small, we instead
#    use the value specified in the widths file. These values are chosen based on our experience as being a good
#    balance broadly sampling parameter space but not being so narrow important solutions are missed. There are two
#    ways a value is specified using the priors/width file:

#   1) Absolute value - 'a' - In this case, the error assumed on the parameter is the value given in the config file.
#      For example, for the width on centre_0 of a light profile, the config file reads centre_0 = a, 0.05. This means
#      if the error on the parameter centre_0 was less than 0.05 in the previous phase, the sigma of its GaussianPrior
#      in this phase will be 0.05.

#   2) Relative value - 'r' - In this case, the error assumed on the parameter is the % of the value of the best-fit
#      value given in the config file. For example, if the intensity estimated in the previous phase was 2.0, and the
#      relative error in the config file is specified as intensity = r, 0.5, then the sigma of the GaussianPrior will be
#      50% of this best-fit value, i.e. sigma = 0.5 * 2.0 = 1.0.

# We use absolute and relative values for different parameters, depending on their properties. For example, using the
# relative value of a parameter like the profile centre makes no sense. If our lens galaxy is centred at (0.0, 0.0),
# the relative error will always be tiny and thus poorly defined. Therefore, the default configs in PyAutoLens use
# absolute errors on the centre.

# However, there are parameters where using an absolute value does not make sense. Intensity is a good example of this.
# The intensity of an image depends on its units, S/N, galaxy brightness, profile definition, etc. There is no single
# absolute value that one can use to generically link the intensity of any two proflies. Thus, it makes more sense
# to link them using the relative value from a previous phase.

### EXAMPLE ##

# Lets go through an example using a real parameter. Lets say in phase 1 we fit the lens galaxy's light with an
# elliptical Sersic profile, and we estimate that its sersic index is equal to 4.0 +- 2.0. To pass this as a prior to
# phase 2, we would write:

def pass_priors(self, previous_results):

    self.lens_galaxies.lens.light.sersic_index = previous_results[0].variable.lens.light.sersic

# The prior on the lens galaxy's sersic light profile would thus be a GaussianPrior in phase 2, with mean=4.0 and
# sigma=2.0.

# If the error on the Sersic index in phase 1 had been really small, lets say, 0.01, we would use the value of the
# Sersic index width in the priors/width config file to set sigma instead. In this case, the prior config file specifies
# that we use an absolute value of 0.8 to link this prior. Thus, the GaussianPrior in phase 2 would have a mean=4.0 and
# sigma=0.8.

# If the prior config file had specified that we use an relative value of 0.8, the GaussianPrior in phase 2 would have
# a mean=4.0 and sigma=3.2.

# And with that, we're done. Linking priors is a bit of an art form, but one that tends to work really well. Its true
# to say that things can go wrong - maybe we 'trim' out the solution we're looking for, or underestimate our errors a
# bit due to making our priors too narrow. However, in general, things are okay, the point is that you should test
# pipelines with different settings, and settle on a setup that appears to be give consistent results but the faster
# run times.