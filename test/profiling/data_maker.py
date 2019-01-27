from test.profiling import makers

import os

# Welcome to the PyAutoLens profiling suite data maker. Here, we'll make the suite of data that we use to profile
# PyAutoLens. This consists of 20 images, which are created as follows:

# 1) A source-only image, where the source galaxy is smooth.
# 2) A source-only image, where the source galaxy is cuspy.
# 3) A lens + source image, where the source galaxy is smooth.
# 4) A lens + source image, where the source galaxy is cuspy.

# Each image is generated at 5 resolutions, 0.2" (LSST), 0.1" (Euclid), 0.05" (HST), 0.03" (HST), 0.01" (Keck AO).

# To simulate each lens, we pass it a name and call its maker. In the makers.py file, you'll see the
makers.make_no_lens_source_smooth(sub_grid_size=16)
makers.make_no_lens_source_cuspy(sub_grid_size=16)
makers.make_lens_and_source_smooth(sub_grid_size=16)
makers.make_lens_and_source_cuspy(sub_grid_size=16)