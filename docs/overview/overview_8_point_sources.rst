.. _overview_8_point_sources:

Point Sources
=============

The overview examples so far have shown strongly lensed galaxies, whose extended surface brightness is lensed into
the awe-inspiring giant arcs and Einstein rings we see in high quality telescope imaging. There are many lenses where
the background source is not extended but is instead a point-source, for example strongly lensed quasars and supernovae.

For these objects, it is invalid to model the source using light profiles, because they implicitly assume an extended
surface brightness distribution. Point source modeling instead assumes the source has a (y,x) `centre`, but
does not have other parameters like elliptical components or an effective radius.

The ray-tracing calculations are now slightly different, whereby they find the locations the point-source's multiple
images appear in the image-plane, given the source's (y,x) centre. Finding the multiple images of a mass model,
given a (y,x) coordinate in the source plane, is an iterative problem that is different to evaluating a light profile.

This example introduces the `PointSolver` object, which finds the image-plane multiple images of a point source by
ray tracing triangles from the image-plane to the source-plane and calculating if the source-plane (y,x) centre is
inside the triangle. The method gradually ray-traces smaller and smaller triangles so that the multiple images can
be determine with sub-pixel precision.

This makes the analysis of strong lensed quasars, supernovae and other point-like source's possible. We also discuss
how fluxes can be associated with the point-source and time delay information can be computed.

The following overview notebook gives a complete run through of point source modeling:

https://github.com/Jammy2211/autolens_workspace/blob/release/notebooks/overview/overview_8_point_sources.ipynb
