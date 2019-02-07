# bisection_minimization.py
# JMA 30 Nov 2018
# coding: utf-8

# # One dimensional line search via estimation
# 
# A robust optimization technique that assumes the function to be optimized 
# is continuous, but not differentiable, and noisy. The function is approximated 
# by a parabola over the set of points that have been computed, assuming that the 
# function is convex withn the range of points

# Since JavaScript output is disabled in JupyterLab you will either need to run this 
# in Jupyter notebooks (JupyterHub) to get plots, or check out the 
# JupyterLab extensions at https://github.com/bokeh/jupyterlab_bokeh
# 
# Example of optimization by bisection line search
import os, sys
import math
import bisect
import pprint
import random
import numpy as np 
import scipy.stats as ss      # scipy.stats.t distribution, e.g. t.cdf

from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import Arrow, NormalHead
from bokeh.palettes import viridis
#from bokeh.io import output_notebook
# output_notebook()


class OneDimOpt:
    'Use a quadratic approx to a set of points in one dimension to search for a minimum'

    # =1: some tracing.  =2: more tracing.  Set to zero to only report warnings. 
    DBG_LVL = 1

    # Search modes
    OK = 0
    NARROW = 1
    WIDEN = 2
    CONVEX = 3
    CONCAVE = 4       # convex up
    LEFTWARD = 5      # lower to the left (pos slope)
    RIGHTWARD = 6     # lower to the right (neg slope)
    CLR_CYCLE = 20


    def __init__(self,
        range_min  = -1, 
        range_max =  1,
        bound_min = None,
        bound_max = None):
        # Initialization 
        self.range_min = range_min
        self.range_max = range_max
        # Default will be to set the search bounds wide. Or optionally to the initial range
        if bound_max is not None:
            self.bound_max = bound_max
        else:
            self.bound_max = sys.float_info.max 
        if bound_min is not None:
            self.bound_min = bound_min
        else:
            self.bound_min = -sys.float_info.max 
        
        self.EPSILON = 1.0e-5


    def init_grid(self):
        'Create an empty sample. '
        self.converge_flag = True
        self.widen_attempts = 0
        self.active_min = float(self.range_min)
        self.active_max = float(self.range_max)
        # tuples of (x, f(x)), ordered by x.
        self.search_grid = []
        # tuples of current estimate & residual avg error, 
        self.residuals = []
        # Also assign plotting colors to points
        self.colors_grid = []
        self.iseq = 0
        self.spectrum = viridis(self.CLR_CYCLE)  # colors to cycle thru. 
        self.spectrum.reverse()
        self.y = []
        self.half_range = (self.range_max - self.range_min)/2

    def init_pts(self, no_pts=3, f = (lambda x:1.0 - x + 10.0*x*x + np.random.random_sample()),
        init_color = 'navajowhite'):
        'Add initial points to the sample.'
        self.f = f
        grid_x = np.linspace(self.range_min, self.range_max, no_pts)
        self.search_grid = list(zip(grid_x, map(self.f, grid_x)))
        self.x = [z[0] for z in self.search_grid]
        self.y = [z[1] for z in self.search_grid]
        self.colors_grid = [init_color] * len(self.search_grid)
        # We want the x value for the min y in the initial sample
        min_at = self.y.index(min(self.y))
        self.initial_guess =  self.x[min_at]
        if self.DBG_LVL > 0:
            print("Initial points:")
            pprint.pprint(self.search_grid)


    def add_f_to_grid(self, x):
        'Add points in sorted order to the sample'
        self.x = [z[0] for z in self.search_grid]
        found_k = bisect.bisect_left(self.x, x)
        # Check if the point goes just after the last point or 
        # if there's not already a point in search grid at x. 
        if (found_k == len(self.x)) or (self.x[found_k] != x):
            new_pt = (x, self.f(x))
            if self.DBG_LVL > 0:
                print('Adding point at ({:.4}, {:.4})'.format( new_pt[0], new_pt[1]))
            self.search_grid.insert(found_k, new_pt)
            self.x = [z[0] for z in self.search_grid]
            self.y = [z[1] for z in self.search_grid]
            self.colors_grid.insert(found_k, self.spectrum[self.iseq % self.CLR_CYCLE])
            self.iseq += 1
        return self.search_grid

    def points_design_matrix(self):
        'The quadratic regression design matrix'
        # Use a quadratic regression to estimate the minimum of the function
        # By centering the data around 0 the columns of the design matrix form an orthogonal basis function
        # Returns the design matrix
        def dm_sample(the_pt):
            'From an x,y tuple build a design matrix row'
            x = the_pt[0]
            return (1, x, x*x)
        dm = np.empty((0,3),dtype='float')
        for row in self.search_grid:
            dm = np.vstack((dm, np.array(dm_sample(row), dtype='float')))
        return dm

    def fit_parabola_to_sample(self, test_size = 1):
        'Use conventional least squares to fit to the design matrix. test_size determines p-value'
        X = self.points_design_matrix()
        fit = {}
        # The fit to the search grid points
        #try:
            # For outputs, see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
        parabola_coef, resid, rank,_ = np.linalg.lstsq(X,self.y, rcond=-1)
        fit["COEFFICIENTS"] = parabola_coef
        fit['RESIDUALS'] = resid
        fit['RANK'] = rank
        # resid is the sum of residuals^2
        # To get the Residual Standard Error. Note the design matrix has 3 degrees of freedom
        fit['RESID_VAR'] = float(resid)/(len(self.y) -3)
        fit['STD_ERR']= self.confidence_bounds(X, test_size * fit['RESID_VAR'])
        #except:
        #    print("WARN: Quadratic fit did not converge {:.5}".format(0.0), file = sys.stderr)
        return fit


    def confidence_bounds(self, X, confidence_factor = 1.0):
        'For testing if a min exists, or is out of bounds using fit standard errors. Returns the std err for a, b, c.'
        inv_mat_diag = np.diag(np.linalg.inv(np.matmul(np.transpose(X),X)))
        inv_mat_diag = [confidence_factor*z for z in inv_mat_diag]
        if self.DBG_LVL > 0:
            print('Std err c: {:.4}, b: {:.4}, a:{:.4}'.format(*inv_mat_diag))  # Check the order. 
        return inv_mat_diag



    def eval_fit(self, coeffs):
        'Compute the fitted values for the parabola fit, along with errors and est min.'
        pts = self.search_grid
        c = coeffs[0]
        b = coeffs[1]
        a = coeffs[2]
        x_s = [z[0] for z in pts]
        y_s = [z[1] for z in pts]
        y_est = [a * x * x + b * x + c for x in x_s]
        # List the absolute value (L1) errors for the fit.
        errs = sum([abs(pt[1] - pt[0]) for pt in zip(y_est, y_s)])/len(y_s)
        print('\tResidual mean abs error: {:.5}'.format(errs))
        if a != 0:
            # Compute estimated minimum
            self.est_min = -0.5*b/a
        else:
            self.est_min = math.nan
        print('\tEstimated min at: {:.4}/2*{:.4} = {:.4}'.format(-b, a, self.est_min))
        self.residuals.append((self.est_min, errs))
        self.est_pts = [x_s, y_est]
        return self


    def new_sample_pt(self):
        'Pick a new point in the current active range biased toward influential values.'
        ### Add a point that straddles the est min, within the active interval
        #   Just using the current min as the new point, biases the converged value
        #   to the current estimate, so this sampling approach avoids this
        #   while still narrowing the sample estimates toward convergence. 
        #   (Might be worth checking that the current estimate stays in the 
        #    active interval.)
        # TODO - need a better way to pick points in the interval. e.g SOBOL randomization,
        # assuming a deterministic function
        return  random.uniform(self.active_min, self.active_max)
        

    def check_fit(self, fit):
        'Is the fit significantly concave?'
        #TODO -if the points check out to be linear, this corner case should not 
        #drive the estimated x to -+ inf. 
        # A concave function will have a negative curvature; std error of the coef is always positive. 
        if fit['COEFFICIENTS'][2] < -fit['STD_ERR'][2]:
            if self.DBG_LVL > -1:
                 print("WARN: Non-convex fit {:.4}: ".format(fit['COEFFICIENTS'][2]), file=sys.stderr)
            return self.CONCAVE
        else:
            return self.OK


    def chose_search(self, fit, max_expansions= 6):
        'Depending on the fit, either expand, narrow or just increase the sample'
        # If min lies outside current range, expand to that side.
        if self.widen_sample(max_expansions):
            self.widen_attempts += 1
        if abs(fit['COEFFICIENTS'][1]) > fit['STD_ERR'][1]: # The slope is significant either way        
            self.widen_symmetric(max_expansions)
            if self.DBG_LVL > -1:
                 print("Significant slope: {:.4}: ".format(fit['COEFFICIENTS'][1]), file=sys.stderr)
        # If convex is not significant and not significant slope, expand in both directions.
        # If significant slope and not significant convex, expand to that side. 
        
         # If high noise, add sample biased toward extremes, and don'e shrink

        # If low noise, shrink extremes. 

    def widen_symmetric(self, max_expansions):
        'Expand in both directions'
        pass

    def widen_sample(self, max_tries = 6):
        'If the estimated min falls out of the range, adjust the range by doubling it to that side.'
        widened = False
        if self.widen_attempts < max_tries:
            if not np.isnan(self.est_min):
                if self.est_min < self.active_min:
                    self.active_min = max(2 * self.active_min - self.active_max, self.bound_min)
                    if self.DBG_LVL > 0:
                        print("\tMin reduced to: ", self.active_min)
                    widened = True
                if self.est_min > self.active_max:
                    self.active_max = min(2 * self.active_max - self.active_min, self.bound_max)
                    if self.DBG_LVL > 0:                
                        print ("\tMax increased to:", self.active_max)
                    widened = True

        else:
            print("WARN:  Exceeded maximum number of interval widenings.", file = sys.stderr)
        return widened

        
    def run_to_convergence(self, max_iterations):
        'Assuming min is in search range, iterate to a fixed point'
        def remove_extreme_pt(pts):
            'An extreme point is the largest of either the first or last of the sample'
            # Remove the first point
            if pts[0][1] > pts[-1][1]:
                if self.DBG_LVL > -1:
                    print("Removed ({:.4}, {:.4})".format( pts[0][0], pts[0][1]))
                del(pts[0])
                del(self.colors_grid[0])
                # Reset the interval extent
                self.active_min = pts[1][0]
            # Remove the last point
            else:
                if self.DBG_LVL > -1:
                    print("Removed ({:.4}, {:.4})".format( pts[-1][0], pts[-1][1]))
                del(pts[-1])
                del(self.colors_grid[-1])
                # Reset the interval extent
                self.active_max = pts[-2][0]
            self.x = [z[0] for z in pts]
            self.y = [z[1] for z in pts]
            return pts
        
        def low_noise():
            return False
                
        not_converged = True
        k = 0
        last_est_min = self.initial_guess
        while not_converged:
            print("\nk = {}".format(k))
            fit = self.fit_parabola_to_sample()
            self.eval_fit(fit['COEFFICIENTS'])
            check= self.check_fit(fit) 
            if check == self.CONCAVE:
                print('WARN: Try a different starting sample.')
                self.converge_flag = False
                break 
            # Has the search converged?
            if abs(self.est_min - last_est_min) < self.EPSILON * self.half_range:
                print('Converged at {:.5}'.format(self.est_min))
                self.converge_flag = True
                break
            if k >= max_iterations:
                print('Exceeded max interations {}'.format(k))
                self.converge_flag = False
                break
            # Check self.est_min 
            # and decide to shrink or to expand it. 
            self.chose_search(fit)
            if self.DBG_LVL > 0:
                print("\tActive interval: [{:.4}, {:.4}]".format (self.active_min, self.active_max))
            # Increase the sample
            active_x = self.new_sample_pt()
            self.add_f_to_grid(active_x)
            # and remove an extreme point. But, only if the noise is low.
            if low_noise():
            # Successively narrow_sample_to_converge by removing eoints far away.'
                self.search_grid = remove_extreme_pt(self.search_grid)

            last_est_min = self.est_min
            k += 1
        return self.search_grid


    def search_for_min(self, max_iterations = 10, target_function = (lambda x: x*x + x),
            initial_sample =5):
        # Create some widely-spaced starting points, to broaden search over possible local optima. 
        self.init_grid()
        self.init_pts(initial_sample, f= target_function)
        self.run_to_convergence(max_iterations = max_iterations )
        return self

##################################################################################################
# Utility functions

def plot_search_grid(search_grid, 
        parabola_pts,
        colors = None):
    p = figure(plot_width = 600, plot_height = 600, 
        title = 'Current points',
        x_axis_label = 'x', y_axis_label = 'f(x)')
    pts = list(zip(*search_grid))
    if colors:

        p.circle(pts[0], pts[1],  color = colors, size=6)
    else:

        p.circle(pts[0], pts[1],  color = colors, size=6)
    # Add a parabola approx.
    p.line(parabola_pts[0], parabola_pts[1], color = 'lightblue')
    return p
    #show(p)


def plot_residuals(residuals):
    p = figure(plot_width = 600, plot_height = 600, 
        title = 'Convergence Path',
        x_axis_label = 'x-est', y_axis_label = 'mean abs residual')
    pts = list(zip(*residuals))
    p.circle(pts[0], pts[1],  color = 'crimson', size=10)
    tail_pt  = residuals[0]
    for a_pt in residuals[1:]:
        p.add_layout(Arrow(end=NormalHead(size=8, fill_color ='lightcoral'), line_color='lightcoral',
        x_start = tail_pt[0], y_start = tail_pt[1], x_end = a_pt[0], y_end = a_pt[1]))
        tail_pt = a_pt
    return p
    #show(p)

###################################################################################
# Test function examples
def v_func(x, lft =10, rht =4, noise = 0.0):
    min_pt = 1
    kcenter = 0
    fx = noise*np.random.random_sample()
    if abs(x - kcenter) < 1e-4:
        fx += min_pt
    if x < kcenter:
        fx += -lft * x + 1
    if x > kcenter:
        fx += rht * x + 1
    return fx 

# A min outside the search range
def almost_lin( x, b = 0, a = -0.01, noise = 1.0):
    return 1 + b*x + a*x*x + noise*np.random.random_sample()
    
# A decidedly non-parabolic function with a global min 
def example_f(x, sc = 2.60,  noise = 0.0010, wave=0.6):
    'Function over the range to minimize in one dim'
    return sc*sc*math.exp((x-1.5)*sc) + sc*sc*math.exp(-(1.9 + x)*sc) + wave* math.sin(4* x) + noise*np.random.random_sample()

################################################################################
### MAIN
################################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1: # input initial guess on from the cmd line
        init_start = float(sys.argv[1])
    else:
        init_start = 10.0

    opt = OneDimOpt(range_min = -1, range_max= 1)
    opt.search_for_min(initial_sample= 11,  max_iterations = 10, target_function = almost_lin)

    sg = plot_search_grid(opt.search_grid, opt.est_pts, opt.colors_grid) #bokeh.palettes.Viridis11) # opt.colors_grid)
    rs = plot_residuals(opt.residuals)  
    show(column(sg, rs))


 
