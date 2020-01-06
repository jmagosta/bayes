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
from scipy.stats import t

from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import Arrow, NormalHead
from bokeh.palettes import viridis

DBG_LVL = 2
WINDOW_LEN = 200  # Training sample size, minutes
WINDOW_START = 100 # Beginning of the sliding window 
HORIZON = 40 # Prediction interval

class scorePredictions(object):
    ''
    def __init__(self, predicted, actual):
        self.n = len(predicted)
        self.diff = predicted - actual

class tsFeatures(object):
    ''
    def __init__(self, src_file):
        self.test = pd.read_csv(src_file, header=0) # TODO for testing, if the actual is avail, plot it also. 
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
    def __init__(self, 
                features, 
                y,
                window_len = WINDOW_LEN,
                window_start = WINDOW_START,
                horizon = HORIZON):
        'X - np array design matrix,  y - data vector.'
        # model train
        self.X = features[window_start:(window_start+window_len),:] 
        self.y = y[window_start:(window_start+window_len)]
        # prediction range
        self.x_prediction = features[(window_start+window_len):(window_start+window_len+horizon),1]
        self.y_prediction = y[(window_start+window_len):(window_start+window_len+horizon)]
        # self.window_len = WINDOW_LEN
        # self.window_start = WINDOW_START
        # self.horizon = HORIZON
        self.n = WINDOW_LEN
        self.mean_x = np.mean(self.X[:,1])
        self.var_x = np.var(self.X[:,1])

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
        try:
            # For outputs, see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
            parabola_coef, resid, rank, sv = np.linalg.lstsq(self.X, self.y, rcond=-1)
            fit["COEFFICIENTS"] = parabola_coef  # "c, b, a"
            fit['RESIDUALS'] = resid  # 2-norm of the residuals
            fit['RANK'] = rank
            # resid is the sum of residuals^2
            # To get the Residual Standard Error - the unbiased estimate of sigma-squared
            # Note the design matrix has 3 degrees of freedom
            fit['RESID_VAR'] = float(resid)/(len(self.y) -rank)
            fit['STD_ERR']= [ test_size * z for z in self.confidence_bounds(self.X, fit['RESID_VAR'])]
        except:
            print("WARN: Quadratic fit did not converge {:.5}".format(0.0), file = sys.stderr)
        return fit

    def predict(self, coeffs):
        y_est = quadratic(self.X[:,1], coeffs)
        y_predict = quadratic(self.x_prediction, coeffs)
        return (pd.DataFrame(dict(x=self.X[:,1], y=self.y, y_est=y_est)), 
                pd.DataFrame(dict(x=self.x_prediction, y=self.y_prediction, y_est=y_predict)))

    def prediction_interval(self, fit, at_x):
        # TODO adjust n for rank. 
        interval_var = fit['RESID_VAR'] *  (1 + (1 + pow((at_x - self.mean_x),2)/self.var_x)/self.n) # TODO - should inc more as at_x gets large.
        # p_interval = t.pdf(self.n-fit['rank']) TODO Need the real t distribution instead of multiplying by 2. 
        p_interval = 2* np.sqrt(interval_var)
        return p_interval

    def upper_prediction(self, pts, y_pred, fit):
        up = [y + self.prediction_interval(fit, x) for x, y in zip(pts, y_pred)]
        return up

##################################################################################################
# Utility functions

def quadratic(pts, coeffs):
    'Compute the fitted values for the parabola fit, along with errors and est min.'
    c = coeffs[0]
    b = coeffs[1]
    a = coeffs[2]
    y_est = [a * x * x + b * x + c for x in pts]
    return y_est



def plot_search_grid(the_estimate, the_prediction, the_upper):
    'Plot a 2 col dataframe'
    p1 = figure(plot_width = 600, plot_height = 600, x_range = (min(the_estimate['x'].values), max(the_prediction['x'].values)),
        title = 'Current points',
        x_axis_label = 'x', y_axis_label = 'f(x)')
    p1.line(the_prediction['x'].values, the_upper, color='red')
    p1.scatter(the_prediction['x'].values, the_prediction['y'].values, color='green', size=2.0)   
    p1.line(the_prediction['x'].values, the_prediction['y_est'].values, color='green')
    p1.scatter(the_estimate['x'].values, the_estimate['y'].values, color='grey', size=2.0) #,alpha=0.4)
    p1.line(the_estimate['x'].values, the_estimate['y_est'].values)
    return p1

 ### MAIN
################################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1: # input sample size
        WINDOW_LEN = int(sys.argv[1])

    input_file = list(Path('.').glob('*pattern.csv'))[0]
    features = tsFeatures(input_file)
    # For n rolling windows of duration d
    prediction = fitAndPredict(features.design_matrix, features.y)
    the_fit = prediction.fit()
    print(the_fit)
    the_estimate, the_prediction = prediction.predict(the_fit["COEFFICIENTS"] )
    x_upper = prediction.upper_prediction(the_prediction.x, the_prediction.y_est, the_fit)
    p = plot_search_grid(the_estimate, the_prediction, x_upper)
    show(p)

