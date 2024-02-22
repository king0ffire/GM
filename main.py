import igl
import scipy as sp
import numpy as np
from meshplot import plot, subplot, interact

import os
root_folder = os.getcwd()
V = np.array([
    [0., 0, 0],
    [1, 0, 0],
    [1, 1, 1],
    [2, 1, 0]
])

F = np.array([
    [0, 1, 2],
    [1, 3, 2]
])
igl.facet_components()
plot(V,F)
igl.per_corner_normals(V,F,90)