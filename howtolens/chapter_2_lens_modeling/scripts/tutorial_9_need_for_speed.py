# %%
"""
Tutorial 9: Need For Speed
==========================

We can now model strong lenses. We can balance complexity and realism to ensure that we infer a good lens model. And
we can do this to make the analysis run faster. However, we always need to be wary of run-time. If we don't craft our
phases carefully, we could spend days, or longer, modeling just one image. In this exercise, we'll get thinking about
what determines the length of time a **PyAutoLens** analysis takes, and how one might speed it up.

__Searching Non-linear Parameter Space__

The more complex your parameter space (e.g. the more parameters), the longer it takes to search. The broader your
priors, the longer it takes to search. The more detailed your non-linear search, the longer it takes. Performing fast
and efficient lens requires us to parameterize and search initial simple non-linear parameterizes, gradually increasing
the complexity of the model whilst exploiting prior linking to keep the run times fast. This will be the main topic of
chapter 3 of the **HowToLens** lectures.

__Algorithmic Optimization__

Every operation AutoLens performs to fit a lens takes time. Every `LightProfile` intensity. Every `MassProfile`
deflection angle. Convolving a model-image with a PSF can take a huge amount of time. As anyone who`s written code
before knows, the better the algorithm is written, the fast it`ll run.

I often get asked, given that **PyAutoLens** is written in Python, isn't it really slow? Afterall, Python is notoriously
slow. Well, no, it isn't. We use a library called `Numba`, that allows us to recompile our Python functions into C
functions before **PyAutoLens** runs. This gives us C-like speed, but in Python code. If you`ve got your own code that needs
speeding up, I strongly recommend that you look up Numba:

http://numba.pydata.org/

We've worked very hard to `Profile` every line of code in **PyAutoLens** and we`re confident its as fast, if not faster,
than any code written in C. In fact, we know this - I wrote the original version of AutoLens in Fortran (bless my
poor soul) and we timed it against **PyAutoLens**. After invoking the magic of Numba, **PyAutoLens** ran 3 times faster than
the Fortran code - I felt pretty smug at that point.

We probably arn`t going to see much more of speed-up via optimization then. Of course, if you`d like to prove me
wrong, go for it - I`ll buy you a beer at a conference someday if you can optimize any function in **PyAutoLens** better
than me!

__Data Quantity__

The final factor driving run-speed is the quantity of data that is being modeled. For every image-pixel that we fit,
we have to compute the `LightProfile` image, the `MassProfile` deflection angles and convolve it with the telescope`s
PSF. The larger that PSF is, the more convolution operations we have to perform too.

In the previous exercises, we used images with a pixel scale of 0.1". I sneakily chose this value cause its fairly
low resolution. Most Hubble images have a pixel scale of 0.05", which is four times the number of pixels! Some
telescopes observe at scales of 0.03" or, dare I say it, 0.01". At these resolutions things can run *really* slow,
if we don't think carefully about run speed beforehand.

Of course, there are ways that we can reduce the number of image-pixels we fit. That`s what masking does. If we
mask out more of the image, we'll fit fewer pixels and **PyAutoLens** will run faster. Alternatively, we could `bin-up`
the image, converting it from say a 0.03" image to a 0.06" image. We lose information, but the code runs faster. We
could trim our PSF to a smaller size, at the risk of modeling our telescope optics worse.

If you want the best, most perfect lens model possible, aggressively masking and cutting the data in this way is a
bad idea. However, if your analysis is composed of multiple phases, and in the early phases you just want a reasonable
lens model which fits the data reasonably well, why bother fitting all the image data? You can do that in the last
phase, right?

Herein lies the beauty behind the pipelines I will introduce in chapter 3. Not only can we tune their navigation of
non-linear parameter space to be fast, we can  freely butcher our data to make **PyAutoLens** run even faster! In the
last phase, we'll fit the complete, unbutchered data-set, so yeah it might take a while to run, but at that point
we've tuned our lens model priors so much the phase should still run reasonably fast.

Therefore, there are no exercises in this tutorial and no code to run. Instead, I just want you to think about how
you might write a pipeline to perform the following analyses:

 1) The only thing you care about is the highly magnified source-galaxy. You don't care about the lens `Galaxy`'s
 `LightProfile`, and its `MassProfile` is only a means to ultimately study the unlensed source. Can you subtract the
 lens `Galaxy`'s light and then discard it in every phase afterwards?

 2) There are 2 lens galaxies responsible for lensing the background source. That means, there are twice as many
 lens galaxy parameters. Can you setup phases that fit each galaxy individiually, before fitting them jointly?

 3) The source galaxy is really complex. Infact, in your strong lens image you count 12 distinct multiple images,
 meaning that there are at least three distinct source`s of light in the source plane. This is a potentially very
 complex non-linear parameter space, so how might you break down the analysis?
"""
