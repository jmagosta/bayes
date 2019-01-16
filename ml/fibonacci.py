# fibonacci.py
# JMA Jan 2019
# coding: utf-8

#  # One dimensional line search by shrinking a determinstic interval 
#
# Textbook method to find the minimum of a continuous convex function 
# See https://www.maplesoft.com/applications/view.aspx?sid=4193&view=html

import os, sys
import math
import bisect
import pprint
import random
import numpy as np 

from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import Arrow, NormalHead
import bokeh.palettes

class SectionOpt:
    'Choose points along a line to successively shrink the search interval.'

    DBG_LVL = 1

    def __init__(self,
        bound_min = None,
        bound_max = None,
        initial_guess = None):
        self.bound_max = bound_max
        self.bound_min = bound_min
        self.initial_guess = initial_guess

################################################################################
### MAIN
################################################################################
if __name__ == "__main__":

    fib = SectionOpt()