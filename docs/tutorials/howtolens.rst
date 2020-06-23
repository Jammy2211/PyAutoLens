.. _howtolens:

HowToLens Lectures
==================

The best way to learn **PyAutoLens** is by going through the **HowToLens** lecture series on the
`autolens workspace <https://github.com/Jammy2211/autolens_workspace>`_.

The lectures are provided as Jupyter notebooks, and they can be browsed on this readthedocs. Of course, I'd recommend
you do them on your own computer by installation **PyAutoFit** and downloading the notebooks.

- **Introduction** - An introduction to strong gravitational lensing & **PyAutolens**.
- **Lens Modeling** - How to model strong lenses, including a primer on Bayesian non-linear analysis.
- **Pipelines** - How to build pipelines & tailor them to your own science case.
- **Inversions** - How to perform pixelized reconstructions of the source-galaxy.
- **Hyper-Mode** - How to use **PyAutoLens** advanced modeling features that adapt the model to the strong lens being analysed.

Config File Path
----------------

If, when running the first notebook, you get an error related to config files, this most likely means that
**PyAutoLens** is unable to find the config files in your autofit workspace. Checkout the
`configs section <https://pyautolens.readthedocs.io/en/latest/general/configs.html>`_ for a description of how to fix this.

Lensing Theory
--------------

**HowToLens** assumes minimal previous knowledge of gravitational lensing and astronomy. However, it is beneficial to
give yourself a basic theoretical grounding as you go through the lectures. I heartily recommend you have open the
lecture course on gravitational lensing by Massimo Meneghetti below as you go through the tutorials, and refer to it
for anything that isn't clear in **HowToLens**.

http://www.ita.uni-heidelberg.de/~massimo/sub/Lectures/gl_all.pdf

Visualization
-------------

Before beginning the **HowToLens** lecture series, in chapter 1 you should do 'tutorial_0_visualization'. This will
take you through how **PyAutoLens** interfaces with matplotlib to perform visualization and will get you setup such
that images and figures display correctly in your Jupyter notebooks.

Jupyter Notebooks
-----------------

The tutorials are supplied as Juypter Notebooks, which come with a '.ipynb' suffix. For those new to Python, Juypter
Notebooks are a different way to write, view and use Python code. Compared to the traditional Python scripts, they
allow:

- Small blocks of code to be viewed and run at a time
- Images and visualization from a code to be displayed directly underneath it.
- Text script to appear between the blocks of code.

This makes them an ideal way for us to present the HowToFit lecture series, therefore I recommend you get yourself
a Juypter notebook viewer (https://jupyter.org/) if you havent done so already.

If you *really* want to use Python scripts, all tutorials are supplied a .py python files in the 'scripts' folder of
each chapter.

For actual **PyAutoLens** use I recommend you use Python scripts. Therefore, as you go through the lecture series you
will notice that we will transition you to Python scripts.

Code Style and Formatting
-------------------------

You may notice the style and formatting of our Python code looks different to what you are used to. For example, it
is common for brackets to be placed on their own line at the end of function calls, the inputs of a function or
class may be listed over many separate lines and the code in general takes up a lot more space then you are used to.

This is intentional, because we believe it makes the cleanest, most readable code possible. In fact, lots of people do,
which is why we use an auto-formatter to produce the code in a standardized format. If you're interested in the style
and would like to adapt it to your own code, check out the Python auto-code formatter 'black'.

https://github.com/python/black


How to Tackle HowToLens
-----------------------

The HowToLens lecture series current sits at 5 chapters, and each will take more than a couple of days to go through
properly. You probably want to be modeling lenses with PyAutoLens faste than that! Furthermore, the concepts in the
later chapters are pretty challenging, and familiarity with PyAutoLens and lens modeling is desirable before you tackle
them.

Therefore, we recommend that you complete chapters 1 & 2 and then apply what you've learnt to the modeling of simulated
and real strong lenses imaging, using the scripts found in the 'autolens_workspace/examples' folder. Once you're happy
with the results and confident with your use of PyAutoLens, you can then begin to cover the advanced functionality
covered in chapters 3, 4 & 5.

Chapter 1: Strong Lensing
-------------------------

In chapter 1, we'll learn about strong gravitational lensing and PyAutoLens. At the end, you'll
be able to:

1) Create uniform grid's of (x,y) Cartesian coordinates.
2) Combine these grid's with light and mass profiles to make images, surface densities, gravitational potentials
   and deflection angle-maps.
3) Combine these light and mass profiles to make galaxies.
4) Perform ray-tracing with these galaxy's whereby a grid is ray-traced through an image-plane / source-plane
   strong lensing configuration.
5) Simulate telescope imaging data of a strong gravitational lens.
6) Fit strong lensing data with model images generated via ray-tracing.
7) Find the model which provides the best-fit to strong gravitational lens imaging.

Chapter 2; Lens Modeling
------------------------

In chapter 2, we'll cover Bayesian inference and non-linear analysis, in particular how we use these tools to
fit imaging of a strong gravitational lens with a lens model. At the end, you'll understand:

1) The concept of a non-linear search and non-linear parameter space.
2) How to fit a lens model to strong lens imaging via  non-linear search.
3) The trade-off between realism and complexity when choosing a lens model.
4) Why an incorrect lens model may be inferred and how to prevent this from happening.
5) The challenges that are involved in inferred a robust lens model in a computationally reasonable run-time.
6) How the results of different lens models can be linked together to perform more efficient and reliable lensing modeling.
7) How different non-linear search algorithms can be used to fit lens models more productively.

Chapter 3: Pipelines
--------------------

In chapter 3, we'll learn how to link ultiple non-linear searches together to build automated lens modeling pipelines.
Once completed, you'll be ready to model your own strong gravitational lenses with PyAutoLens!

With these pipelines, you'll be able to:

1) Fit a lens mass model and source light model to an image of a strongly lensed source.
2) Additionally fit the lens galaxy's light, if it is present.
3) Write customized pipelines for strong lens systems with multiple lens galaxies or source galaxies.
4) Customize pipelines such that the priors on parameters during the fit are adjusted to provide a more robust or
   efficient fit.

Chapter 4: Pixelizations
------------------------

In chapter 4, we'll explain how the lensed source galaxy an be reconstructed using a pixelized grid, meaning that
sources with complex and irregular morphologies can be fitted accurately. You'll learn how to:

1) Pixelize a source-plane into a set of source-plane pixels that define mappings to image pixels.
2) Invert this source-plane pixelization to fit the strongly lensed source and thus reconstruct its light.
3) Apply a smoothness prior on our source reconstruction, called 'regularization', to ensure our solution is physical.
4) Apply this prior in a Bayesian framework to objectively quantify our source reconstruction's log likelihood.
5) Define a border in the source-plane to prevent pixels tracing outside the source reconstruction.
6) Use alternative pixelizations, for example a Voronoi grid whose pixels adapt to the lens's mass model.
7) Use these features in PyAutoLens pipelines.

Chapter 5: Hyper Mode
---------------------

In hyper-mode, we introduced advanced functionality into PyAutoLens that adapts various parts of the lens modeling
procedure to the data that we are fitting.

NOTE: Hyper-mode is conceptually quite challenging, and I advise that you make sure you are very familiar with
PyAutoLens before covering chapter 5!

1) Adapt an inversions's pixelization to the morphology of the reconstructed source galaxy.
2) Adapt the regularization scheme applied to this source to its surface brightness profile.
3) Use hyper-galaxies to scale the image's noise map during fitting, to prevent over-fitting regions of the image.
4) Include aspects of the data reduction in the model fitting, for example the background sky subtraction.
5) Use these features in PyAutoLens's pipeline frame.