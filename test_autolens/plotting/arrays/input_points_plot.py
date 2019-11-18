import autolens as al
import numpy as np

array = al.array.manual_2d(array=np.ones((50, 50)), pixel_scales=0.1)

# al.plot.array(arrays=arrays, centres=[[(1.0, 1.0)], [(-1.0, 1.0)], [(-2.0, -2.0), (-3.0, -3.0)]])
al.plot.array(array=array, centres=[[(0.0, 0.0)]], axis_ratios=[[0.5]], phis=[[45.0]])
stop
al.plot.array(
    array=array,
    centres=[[(0.0, 0.0)], [(-1.0, 1.0)], [(-2.0, -2.0), (-3.0, -3.0)]],
    axis_ratios=[[0.5], [0.5], [0.3, 0.3]],
    phis=[[0.0], [90.0], [45.0, 15.0]],
)
al.plot.array(array=array, points=[[[1.0, 1.0], [2.0, 2.0]], [[-1.0, -1.0]]])
