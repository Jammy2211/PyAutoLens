from autolens.data.array import scaled_array
from autolens.data.array.plotters import array_plotters
import numpy as np

array = scaled_array.ScaledSquarePixelArray(array=np.ones((50, 50)), pixel_scale=0.1)

centres = [(1.0, 1.0), (-1.0, 1.0), (-2.0, -2.0)]

array_plotters.plot_array(array=array, centres=centres)