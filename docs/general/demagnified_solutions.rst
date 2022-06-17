.. _demagnified_solutions:

Demagnified Solutions
=====================

Overview
--------

When fitting a lens model using a pixelized source reconstruction, it is common for unphysical and inaccurate lens
models to be estimated.

This is due to demagnified source reconstructions, where the source is reconstructed as a near identical realisization 
of the lensed source galaxy:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/general/images/data.png
  :width: 400
  :alt: Alternative text

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/general/images/model_image.png
  :width: 400
  :alt: Alternative text

What has happened in the model above is that the mass model has gone to an extremely low mass solution, such that
essentially no ray-tracing occurs. The best source reconstruction to fit the data is therefore one where the source is
reconstructed to look exactly like the observed lensed source.

In fact, there are two variants of these unwanted systematic solutions, with the second variant correspond to mass
models with **too much mass**, such that the ray-tracing inverts in on itself.

The following schematic is from the paper Maresca et al 2021 (https://arxiv.org/abs/2012.04665) and illustrates
this beautifully:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/general/images/maresca_fig1.png
  :width: 400
  :alt: Alternative text

The source reconstructions and model-fits of these solutions are also brilliant illustrated by Maresca et al 2021:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoLens/master/docs/general/images/maresca_fig2.png
  :width: 400
  :alt: Alternative text

The demagnified solutions are not as high likelihood as the physical and accurate mass model shown in the central
row of the figure above. However, the demagnified solutions **occupy a much greater volume of non-linear parameter space**.
This means that when we fit a lens model, it is highly probable that all solutions correspond to demagnified solutions,
that the non-linear search converges on these solutions and never samples the accurate physical mass model.

Solution
--------

The prevent a non-linear search from inferring these unwanted solutions **PyAutoLens** penalizes the likelihood
via a position thresholding term. 

First, we specify the locations of the lensed source's multiple images, which the example code below does for
the sample strong lens pictured below, where multiple images are drawn on with black stars:

.. code-block:: python
    
    positions = al.Grid2DIrregular(
        grid=[(0.4, 1.6), (1.58, -0.35), (-0.43, -1.59), (-1.45, 0.2)]
    )

The ``autolens_workspace`` also includes a Graphical User Interface for drawing lensed source positions via 
mouse click ().

Next, we create ``PositionsLHPenalty`` object, which has an input ``threshold``.

This requires that a mass model traces the multiple image ``positions`` specified above within the ``threshold`` 
value (e.g. 0.5") of one another in the source-plane. If this criteria is not met, a large penalty term is
applied to likelihood that massively reduces the overall likelihood. This penalty is larger if the ``positions``
trace further from one another.

This ensures the unphysical solutions that produce demagnified solutions have a much lower likelihood that the 
physical solutions we desire. Furthermore, the penalty term reduces as the image-plane multiple image ``positions`` trace 
closer in the source-plane, ensuring the non-linear search (E.g. Dynesty) converges towards an accurate mass model. 
It does this very fast, as ray-tracing just a few multiple image positions is computationally cheap. 

If the ``positions`` do trace within the ``threshold`` no penalty is applied.

The penalty term is created and passed to an ``Analysis`` object as follows:

.. code-block:: python

    positions_likelihood = al.PositionsLHPenalty(positions=positions, threshold=0.3)

    analysis = al.AnalysisImaging(
        dataset=imaging, positions_likelihood=positions_likelihood
    )

The threshold of 0.5" is large. For an accurate lens model we would anticipate the positions trace within < 0.01" of
one another. However, we only want the threshold to aid the non-linear with the choice of mass model in the intial fit
and remove demagnified solutions.

Resampling
----------

An alternative penalty term is available via the ``PositionsLHResample`` object, which rejects and resamples a lens
model if the ``positions``do not trace within the ``threshold`` of one another in the source plane.

We do not recommend users use this, as it is slower and can often lead to prolonged periods of the non-linear search
guessing and rejecting mass models.

.. code-block:: python

    positions_likelihood = al.PositionsLHResample(positions=positions, threshold=0.3)

    analysis = al.AnalysisImaging(
        dataset=imaging, positions_likelihood=positions_likelihood
    )