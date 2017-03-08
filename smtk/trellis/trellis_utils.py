#!/usr/bin/env python
# LICENSE
#
# Copyright (c) 2010-2014, GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
#
# The Hazard Modeller's Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>
"""
Trellising Utilities
"""

import numpy as np
from math import sqrt, ceil

def best_subplot_dimensions(nplots):
    """
    Returns the optimum arrangement of number of rows and number of
    columns given a total number of plots. Converted from the Matlab function
    "BestArrayDims" 
    :param int nplots:
        Number of subplots
    :returns:
        Number of rows (int)
        Number of columns (int)
    """
    if nplots == 1:
        return 1, 1
    nplots = float(nplots)
    d_l = ceil(sqrt(nplots))
    wdth = np.arange(1., d_l + 1., 1.)
    hgt = np.ceil(nplots / wdth)
    waste = (wdth * hgt - nplots) / nplots
    savr = (2. * wdth + 2. * hgt) / (wdth * hgt)
    cost = 0.5 * savr + 0.5 * waste
    num_col = np.argmin(cost)
    num_row = int(hgt[num_col])
    return num_row, num_col + 1
