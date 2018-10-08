# In the previous 3 pipelines, we passed priors using the 'variable' attribute of the previous results. However, its
# not yet clear how these priors are passed. Do they use a UniformPrior or GaussianPrior? What are the liimts / sigma
# on these priors? Can I change this behaviour?

# Lets say I link two parameters in pass priors (don't run this code its just a demo)

def pass_priors(self, previous_results):

    self.lens_galaxies[0].profile.parameter = previous_results[0].variable.lens_galaxies.profile.parameter

# By invoking the 'variable' attribute, the passing of priors behaves following 3 rules:

# 1) The ' self.lens_galaxies[0].profile.parameter' parameter will use a GaussianPrior as its prior.
#
#    A GaussianPrior is ideal, as the 1D pdf results we compute at the end of a phase are easily summarized as
#    Gaussians.

# 2) The GaussianPrior mean is the best-fit value of 'previous_results[0].variable.lens_galaxies.profile.parameter'.
#
#    As you'd guess, this means our prior starts searching the region of non-linear parameter space that correspond to
#    highest likelihood solutions in the previous phase.

# 3) The sigma of the Gaussian will use either: (i) the 1D error on the previous result's parameter or; (ii) the value
#    specified in the appropriate 'config/priors/width/profile.ini' config file (check these files out now).

#    The idea here is simple. We want a value of sigma that gives a GaussianPrior wide enough to search a broad region
#    of parameter, so that the lens model can change if a better solution is nearby. However, we want it to be narrow
#    enough that we don't search too much of parameter space, or risk getting lost! A natural choice is the errors on
#    that parameter in the previous phase.

#    Unfortunately, this doesn't always work. Lens modeling is prone to an effect called 'over-fitting' where we
#    underestimate the errors on our lens model parameters. This is especially true when we take the shortcuts we're
#    used to in early phases - aggresive masking, reduced data, simplified lens models.

#    Therefore, the priors/widths file is our fallback. If the error on a parameter is suspiciously small, we instead
#    use the value specified in the widths file. These values are chosen based on our experience as being a good
#    balacne broadly sampling parameter space but not being so narrow important solutions are missed.


# There's a couple  more thing we should think about. You might be worried that we're always at risk of not specifying
# broad enough priors, and that we'll end up trimming out the best-fit solution to our lens model. This can happen -
# balancing broadness and narrowness is a bit of an art form.

# You may also be worried that our priors will impact our inferred errors. If our priors are narrow, the errors that
# we infer will, you guessed it, also be quite narrow. They might not be representative of the 'true errors' - and
# as a scientist we want true errors!

# In light of this, I recommend that a pipeline's final phase is a phase where we take a hit on run-time, and choose
# settings that more thoroughly sample non-linear parameter space. This includes larger values of sigma on each
# GaussianPrior, and upping the MultiNest live points / reducing its sampling efficiency. In truth, the degree to which
# this matters depends on your lens model complexity, data-quality and science case, so it's something you should
# learn how to do yourself.