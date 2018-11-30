# We can now model strong lenses. We can balance complexity and realism to ensure that we infer a good lens model. And
# we can do this to make the analysis run faster. However, we always need to be wary of run-time. If we don't craft
# our phases carefully, we could spend days, or longer, modeling just one regular. In this exercise, we'll get thinking
# about what determines the length of time an AutoLens analysis takes, and how one might speed it up.

### Searching Non-linear Parameter Space ###

# We've covered this already. The more complex your parameter space (e.g. the more parameters), the longer it takes to
# search. The broader your priors, the longer it takes to search. The slower you search non-linear parameter space,
# the longer PyAutoLens takes to run.

# There isn't much else to say here - just that, once we start thinking about pipeline's, we'll start linking phases
# together in ways that navigate non-linear parameter spaces in an extremely efficient manner.

### Algorithmic Optimization ###

# Every operation AutoLens performs to fit a lens takes time. Every light profile intensity. Every mass profile
# deflection angle. Convolving a model-regular with a PSF can take a huge amount of time. As anyone who's written code
# before knows, the better the algorithm is written, the fast it'll run.

# I often get asked, given that PyAutoLens is written in Python, isn't it really slow? Afterall, Python is notoriously
# slow. Well, no, it isn't. We use a library called 'Numba', that allows us to recompile our Python functions into
# C functions before PyAutoLens runs. This gives us C-like speed, but in Python code. If you've got your own code that
# needs speeding up, I strongly recommend that you look up Numba:

# http://numba.pydata.org/

# We've worked very hard to profile every line of code in AutoLens and we're confident its as fast, if not faster,
# than any code written in C. In fact, we know this - I wrote the original version of AutoLens in Fortran (bless my
# soul) and we timed it against PyAutoLens. After invoking the magic of Numba, PyAutoLens ran 3 times faster than the
# Fortran code - I felt pretty smug at that point.

# We probably arn't going to see much more of speed-up via optimization then. Of course, if you'd like to prove me
# wrong, go for it - I'll buy you a beer at a conference someday if you do.

### Data Quantity ###

# The final factor driving run-speed is the shear quantity of datas that we're modeling. For every regular-pixel that we
# fit, we have to compute the light-profile intensities, the mass-profile deflection angles and convolve it with
# the telescope's PSF. The larger that PSF is, the more convolution operations we have to perform too.

# In the previous exercises, we used regular with a pixel scale of 0.1". I sneakily chose this value cause its fairly
# low resolution. Most Hubble regular have a pixel scale of 0.05", which is four times the number of pixels! Some
# telescopes observe at scales of 0.03" or, dare I say it, 0.01", at these resolutions we things will run really slow,
# if we don't think carefully about run speed beforehand.

# Of course, there are ways that we can reduce the number of regular-pixels we fit. That's what masking does. If we masks
# out more of the regular, we'll fit fewer pixels and AutoLens will run faster. Alternatively, we could 'bin-up' the
# regular, converting it from say a 0.03" regular to a 0.06" regular. We lose information, but the code runs
# faster. We could trim our PSF to a smaller size, at the risk of modeling our telescope optics worse.

# If you want the best, most perfect lens model possible, aggressively masking and cutting the datas in this way is a
# bad idea. However, if your analysis is composed of multiple phases, and in the early phases you just want a
# reasonable lens model which fits the datas reasonably well, why bother fitting all the regular datas?
# You can do that in the last phase, right?

# Herein lies the beauty behind runners. Not only can we tune their navigation of non-linear parameter space, we can
# freely butcher our datas to keep AutoLens running fast. Yeah, the last phase might take a while to run, because
# we're fitting a larger quantity of datas, but at this point we've tuned our lens model priors so much the phase can't
# take too long.

# Therefore, there are n o exercises in this tutorial and no code to run. We're going to hold off thinking about
# run-speed until we introduce runners. Instead, I just want you to think about how you might write a pipeline to
# perform the following analyses:

# 1) The only thing you care about is the highly magnified source-model_galaxy. You don't care about the lens model_galaxy's
#    light profile, and its mass-profile is only a means to ultimately study the unlensed source. Can you subtract
#    the lens model_galaxy's light and then discard it in every phase afterwards?

# 2) There are 2 lens galaxies responsible for lensing the background source. That means, there are twice as many
#    lens model_galaxy parameters. Can you setup phases that fit each model_galaxy individiually, before fitting them jointly?

#  3) The source model_galaxy is really complex. Infact, in your strong lens regular you count 12 distinct multiple regular,
#     meaning that there are at least three distinct source's of light in the source plane. This is a potentially
#     very complex non-linear parameter space, so how might you break down the analysis?