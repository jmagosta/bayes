# convex_min_test.py
# JMA 24 March 2019
'''Test file for resource surface minimization 
'''
import pytest
from . line_minimization import OneDimOpt   # Relative import from current directory

class Testalg(object):
    
    def setup(this):
        this.opt = OneDimOpt()
