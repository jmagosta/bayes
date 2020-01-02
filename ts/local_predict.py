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
from pathlib import Path
import numpy as np
import pandas as pd
# import scipy.stats as ss      # scipy.stats.t distribution, e.g. t.cdf

from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import Arrow, NormalHead
from bokeh.palettes import viridis

DBG_LVL = 2
WINDOW_LEN = 30  # minutes

class scorePredictions(object):
    ''
    def __init__(self, predicted, actual):
        self.n = len(predicted)
        self.diff = predicted - actual

class tsFeatures(object):
    ''
    def __init__(self, src_file):
        self.test = pd.read_csv(src_file, header=0)
        # Create feature dataframe
        self.ts = self.test[['minutes', 'cpuload']]
        minutes = self.ts['minutes'].values
        squared = minutes * minutes
        intercept = np.ones(len(minutes))
        self.design_matrix = np.stack((intercept, minutes, squared)).T
        self.y = self.test['cpuload']
        # self.ts = pd.concat
        # Add daily components

        # Add Serial indexes

        # difference the series

class fitAndPredict(object):
    ''
    def __init__(self, features, y):
        'X - np array design matrix,  y - data vector.'
        self.X = features
        self.y = y

    def confidence_bounds(self, X, resid_var):
        'For testing if a min exists, or is out of bounds using fit standard errors. Returns the std err for a, b, c.'
        inv_mat_diag = np.diag(np.linalg.inv(np.matmul(np.transpose(X),X)))
        inv_mat_diag = [math.sqrt(resid_var*z) for z in inv_mat_diag]
        if DBG_LVL > 1:
            print('Std err c: {:.4}, b: {:.4}, a:{:.4}'.format(*inv_mat_diag))
        return inv_mat_diag

    def fit(self, test_size = 1.9):
        fit = {}
        # The fit to the search grid points
        #try:
            # For outputs, see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
        parabola_coef, resid, rank,_ = np.linalg.lstsq(self.X,self.y, rcond=-1)
        fit["COEFFICIENTS"] = parabola_coef
        fit['RESIDUALS'] = resid
        fit['RANK'] = rank
        # resid is the sum of residuals^2
        # To get the Residual Standard Error. Note the design matrix has 3 degrees of freedom
        fit['RESID_VAR'] = float(resid)/(len(self.y) -3)
        fit['STD_ERR']= [ test_size * z for z in self.confidence_bounds(self.X, fit['RESID_VAR'])]
        #except:
        #    print("WARN: Quadratic fit did not converge {:.5}".format(0.0), file = sys.stderr)
        return fit

    def predict(self, coeffs, pts):
        'Compute the fitted values for the parabola fit, along with errors and est min.'
        c = coeffs[0]
        b = coeffs[1]
        a = coeffs[2]
        y_est = [a * x * x + b * x + c for x in pts]
        return pd.DataFrame(dict(x=pts, y=self.y, predict=y_est))


##################################################################################################
# Utility functions

def plot_search_grid(the_estimate,
        colors = None):
    'Plot a 2 col dataframe'
    p1 = figure(plot_width = 600, plot_height = 600, 
        title = 'Current points',
        x_axis_label = 'x', y_axis_label = 'f(x)')
    p1.scatter(the_estimate['x'].values, the_estimate['y'].values, color='grey', size=2.0) #,alpha=0.4)
    p1.line(the_estimate['x'].values, the_estimate['predict'].values)
    return p1

 ### MAIN
################################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1: # input sample size
        WINDOW_LEN = int(sys.argv[1])

    input_file = list(Path('.').glob('*pattern.csv'))[0]
    features = tsFeatures(input_file)
    # For n rolling windows of duration d
    prediction = fitAndPredict(features.design_matrix[0:WINDOW_LEN,:], features.y[0:WINDOW_LEN])
    the_fit = prediction.fit()
    print(the_fit)
    the_estimate = prediction.predict(the_fit["COEFFICIENTS"], prediction.X[:,1] )
    p = plot_search_grid(the_estimate)
    show(p)


    
    