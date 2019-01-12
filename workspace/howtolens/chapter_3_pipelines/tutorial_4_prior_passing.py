# In the previous 3 runners, we passed priors using the 'variable' attribute of the previous results. However, its
# not yet clear how these priors are passed. Do they use a UniformPrior or GaussianPrior? What are the liimts / sigma
# on these priors? Can I change this behaviour?

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
#    corresponded to highest likelihood solutions in the previous phase. Thus, we're setting our prior to look in the
#    correct regioon of parameter space.

# 3) The sigma of the Gaussian will use either: (i) the 1D error on the previous result's parameter or; (ii) the value
#    specified in the appropriate 'config/priors/width/profile.ini' config file (check these files out now).

#    The idea here is simple. We want a value of sigma that gives a GaussianPrior wide enough to search a broad region
#    of parameter, so that the lens model can change if a better solution is nearby. However, we want it to be narrow
#    enough that we don't search too much of parameter space, or risk getting lost! A natural choice is the errors on
#    that parameter in the previous phase.

#    Unfortunately, this doesn't always work. Lens modeling is prone to an effect called 'over-fitting' where we
#    underestimate the errors on our lens model parameters. This is especially true when we take the shortcuts we're
#    used to in early phases - aggresive masking, reduced data, simplified lens models, constant efficiency mode, etc.

#    Therefore, the priors/widths file is our fallback. If the error on a parameter is suspiciously small, we instead
#    use the value specified in the widths file. These values are chosen based on our experience as being a good
#    balance broadly sampling parameter space but not being so narrow important solutions are missed.


# Lets go through an example using a real parameter. Lets say in phase 1 we fit the lens galaxy's light with an
# elliptical Sersic profile, and we estimate that its sersic index is equal to 4.0 +- 2.0. To pass this as a prior to
# phase 2, we would write:

def pass_priors(self, previous_results):

    self.lens_galaxies.lens.light.sersic_index = previous_results[0].variable.lens.light.sersic

# The prior on the lens galaxy's sersic light profile would thus be a GaussianPrior in phase 2, with mean=4.0 and
# sigma=2.0.

# If the error on the Sersic index in phase 1 had been really small, lets say, 0.01, we would use the value of the
# Sersic index width in the priors/width config file to set sigma instead. In this case, the GaussianPrior in phase 2
# would have a mean=4.0 and sigma=0.8.

# And with that, we're done. Linking priors is a bit of an art form, but one that tends to work really well. Its true
# to say that things can go wrong - maybe we 'trim' out the solution we're looking for, or underestimate our errors a
# bit due to making our priors too narrow. However, in general, things are okay, the point is that you should test
# pipelines with different settings, and settle on a setup that appears to be give consistent results but the faster
# run times.