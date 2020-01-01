# local_predict.py
# JMA 1 Jan 2020

# Time series prediction using linear models
#
# Using basic numpy linear algebra as the regressor
import os, sys
import math
import bisect
import pprint
import random
import numpy as np 
# import scipy.stats as ss      # scipy.stats.t distribution, e.g. t.cdf

from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import Arrow, NormalHead
from bokeh.palettes import viridis

class scorePredictions(object):
    pass

class tsFeatures(object):
    pass

class fitAndPredict(object):
    pass

##################################################################################################
# Utility functions

def plot_search_grid(search_grid, 
        parabola_pts,
        colors = None):
    p = figure(plot_width = 600, plot_height = 600, 
        title = 'Current points',
        x_axis_label = 'x', y_axis_label = 'f(x)')

    return p

 ### MAIN
################################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1: # input initial guess on from the cmd line
        init_start = float(sys.argv[1])

    features = tsFeatures()

    
    