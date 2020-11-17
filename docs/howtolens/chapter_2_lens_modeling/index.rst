Chapter 2: Lens Modeling
========================

In chapter 2, we'll take you through how to model strong lenses using a non-linear search.

A number of the notebooks require a `NonLinearSearch` to be performed, which can lead the auto-generation of the
**HowToLens** readthedocs pages to crash. For this reason, all cells which perform a `NonLinearSearch` or use its
result are commented out. We advise if you want to read through the **HowToLens** lectures in full that you download
the autofit_workspace and run them from there (where these comments are removed).

The chapter contains the following tutorials:

`Tutorial 1: <https://pyautolens.readthedocs.io/en/latest/howtolens/chapter_2_lens_modeling/tutorial_1_non_linear_search.html>`_
- How a `NonLinearSearch` is used to fit a lens model.

`Tutorial 2: Parameter Space And Priors <https://pyautolens.readthedocs.io/en/latest/howtolens/chapter_2_lens_modeling/tutorial_2_parameter_space_and_priors.html>`_
- The Concepts of a parameter space and priors.

`Tutorial 3: Realism and Complexity <https://pyautolens.readthedocs.io/en/latest/howtolens/chapter_2_lens_modeling/tutorial_3_realism_and_complexity.html>`_
- Finding a balance between realism and complexity when composing a lens model.

`Tutorial 4: Dealing with Failure <https://pyautolens.readthedocs.io/en/latest/howtolens/chapter_2_lens_modeling/tutorial_4_dealing_with_failure.html>`_
- What to do when PyAutoLens finds an inaccurate lens model.

`Tutorial 5: Linking Phases <https://pyautolens.readthedocs.io/en/latest/howtolens/chapter_2_lens_modeling/tutorial_5_linking_phases.html>`_
- Breaking the lens modeling procedure down into multiple fits.

`Tutorial 6: Alternative Searches  <https://pyautolens.readthedocs.io/en/latest/howtolens/chapter_2_lens_modeling/tutorial_6_alternative_searches.html>`_
- Using different non-linear searches to sample parameter space.

`Tutorial 7: Masking and Positions <https://pyautolens.readthedocs.io/en/latest/howtolens/chapter_2_lens_modeling/tutorial_7_masking_and_positions.html>`_
- How to mask and mark positions on your ``data`` to improve the lens model.

`Tutorial 8: Results <https://pyautolens.readthedocs.io/en/latest/howtolens/chapter_2_lens_modeling/tutorial_8_results.html>`_
- Overview of the results available after successfully fitting a lens model.

`Tutorial 9: Need for Speed <https://pyautolens.readthedocs.io/en/latest/howtolens/chapter_2_lens_modeling/tutorial_9_need_for_speed.html>`_
- How to fit complex models whilst balancing efficiency and run-time.

.. toctree::
   :caption: Tutorials:
   :maxdepth: 1
   :hidden:

   tutorial_1_non_linear_search
   tutorial_2_parameter_space_and_priors
   tutorial_3_realism_and_complexity
   tutorial_4_dealing_with_failure
   tutorial_5_linking_phases
   tutorial_6_alternative_searches
   tutorial_7_masking_and_positions
   tutorial_8_results
   tutorial_9_need_for_speed