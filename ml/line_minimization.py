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

from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import Arrow, NormalHead
import bokeh.palettes
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
    NONCONVEX = 3
    CLR_CYCLE = 20


    def __init__(self,
        range_min  = -1, 
        range_max =  1,
        bound_min = None,
        bound_max = None,
        initial_guess = None):
        # Initialization 
        self.range_min = range_min
        self.range_max = range_max
        self.initial_guess = initial_guess
        # Default will be to set the search bounds wide. Or optionally to the initial range
        if bound_max is not None:
            self.bound_max = bound_max
        else:
            self.bound_max = sys.float_info.max 
        if bound_min is not None:
            self.bound_min = bound_min
        else:
            self.bound_min = sys.float_info.min 
        
        self.EPSILON = 1.0e-5


    def init_grid(self):
        ''
        self.converge_flag = True
        self.active_min = float(self.range_min)
        self.active_max = float(self.range_max)
        # tuples of (x, f(x)), ordered by x.
        self.search_grid = []
        # tuples of current estimate & residual avg error, 
        self.residuals = []
        # Also assign plotting colors to points
        self.colors_grid = []
        self.iseq = 0
        self.spectrum = bokeh.palettes.viridis(self.CLR_CYCLE)  # colors to cycle thru. 
        self.spectrum.reverse()
        self.y = []
        self.half_range = (self.range_max - self.range_min)/2

    def init_pts(self, no_pts=3, f = (lambda x:1.0 - x + 10.0*x*x + np.random.random_sample()),
        init_color = 'navajowhite'):
        self.f = f
        grid_x = np.linspace(self.range_min, self.range_max, no_pts)
        self.search_grid = list(zip(grid_x, map(self.f, grid_x)))
        self.x = [z[0] for z in self.search_grid]
        self.y = [z[1] for z in self.search_grid]
        self.colors_grid = [init_color] * len(self.search_grid)
        if not self.initial_guess:
            # No, we want the x value for the min y. 
            min_at = self.y.index(min(self.y))
            self.initial_guess =  self.x[min_at]
        if self.DBG_LVL > 0:
            print("Initial points:")
            pprint.pprint(self.search_grid)


    def add_f_to_grid(self, x):
        'Add points in sorted order to the sample'
        # Binary search would be better
        found_k = bisect.bisect_left([z[0] for z in self.search_grid],x)
        # Check if the point is after the last point or 
        # if there's not already a point in search grid at x. 
        if found_k > len(self.x) or self.x[found_k] != x:
            new_pt = (x, self.f(x))
            if self.DBG_LVL > 0:
                print('Adding point at ({:.4}, {:.4})'.format( new_pt[0], new_pt[1]))
            self.search_grid.insert(found_k, new_pt)
            self.x = [z[0] for z in self.search_grid]
            self.y = [z[1] for z in self.search_grid]
            self.colors_grid.insert(found_k, self.spectrum[self.iseq % self.CLR_CYCLE])
            self.iseq += 1
        return self.search_grid[found_k]

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

    def fit_parabola_to_sample(self):
        'Use conventional least squares to fit to the design matrix.'
        X = self.points_design_matrix()
        if OneDimOpt.DBG_LVL > 1:
            print('Design matrix:\n', X)
        # The fit to the search grid points
        # For outputs, see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
        parabola_fit = np.linalg.lstsq(X,self.y, rcond=-1)
        # Return the coefficients (c, b a)
        self.quadratic_coeff = parabola_fit[0]
        return self.quadratic_coeff

    def fit_line_to_sample(self):
        'Use conventional least squares to fit to the design matrix.'
        X = self.points_design_matrix()
        X = X[:, (0,1)]
        if OneDimOpt.DBG_LVL > 1:
            print('Design matrix:\n', X)
        # The fit to the search grid points
        linear_fit = np.linalg.lstsq(X,self.y, rcond=-1)
        # Return the coefficients (c, b a)
        self.linear_coeff = (linear_fit[0][0], linear_fit[0][1], 0)
        return self.linear_coeff

    # When the parabola fails, a linear or constant fit may show, by fitting better. 
    def fit_constant(self):
        # y = [z[1] for z in self.search_grid]
        if len(self.y) == 0:
            print("No samples found.", file=sys.stderr )
        y_est = np.mean(self.y)
        self.constant_coeff = (y_est, 0, 0)
        return self.constant_coeff

    # Search and convergence
    def check_imbalance(self):
        'Is one parabola side much higher than the other?'
        pass

    def check_inconsistent_slope(self):
        'Is the linear fit divergent from the quadratic?'
        pass

    def check_convex(self):
        'Is the fit convex?'
        if self.quadratic_coeff[2] < 0:
            if self.DBG_LVL > -1:
                print("WARN: Non-convex fit {:.4}: ", self.quadratic_coeff[2], file=sys.stderr)
            return self.NONCONVEX
        else:
            return self.OK

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
                return pts
            # Remove the last point
            else:
                if self.DBG_LVL > -1:
                    print("Removed ({:.4}, {:.4})".format( pts[-1][0], pts[-1][1]))
                del(pts[-1])
                del(self.colors_grid[-1])
                # Reset the interval extent
                self.active_max = pts[-2][0]
                return pts
        
        def low_noise():
            return True
                
        not_converged = True
        k = 0
        last_est_min = self.initial_guess
        while True:
            print("\nk = {}".format(k))
            pts = self.eval_fit(self.fit_parabola_to_sample())
            if self.check_convex() == self.NONCONVEX:
                print('WARN: Try a different starting sample.')
                self.converge_flag = False
                break 
            if abs(self.est_min - last_est_min) < self.EPSILON * self.half_range:
                print('Converged at {:.5}'.format(self.est_min))
                break
            if k >= max_iterations:
                break
            ### Add a point that straddles the est min, within the active interval
            #   Using the current min as the new point, biases the converged value
            #   to the current estimate, so this sampling approach avoids this
            #   while still narrowing the sample estimates toward convergence. 
            #   (Might be worth checking that the current estimate stays in the 
            #    active interval.)
            # TODO here's where to check if self.est_min is outside the active interval,
            # and decide how to expand it. 
            if self.DBG_LVL > 0:
                print("\tActive interval: [{:.4}, {:.4}]".format (self.active_min, self.active_max))

            # TODO - need a better way to pick points in the interval. e.g SOBOL randomization,
            # assuming a deterministic function

            active_x = random.uniform(self.active_min, self.active_max)
            self.add_f_to_grid(active_x)
            # and remove an extreme point. But, only if the noise is low.
            if low_noise():
                self.search_grid = remove_extreme_pt(self.search_grid)
            self.y = [z[1] for z in self.search_grid]
            last_est_min = self.est_min
            k += 1
        return self.search_grid

    def eval_fit(self, coeffs):
        'Compute the fitted values for the parabola fit'
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
        return [x_s, y_est]
        
    def narrow_sample_to_converge(self, max_iterations = 10, target_function = (lambda x: x*x + x),
            initial_sample =5):
        'Successively narrow_sample_to_converge by adding points near the estimated min, and remove points far away.'
        # Create some widely-spaced starting points, to broaden search over possible local optima. 
        self.init_grid()
        self.init_pts(initial_sample, f= target_function)
        self.run_to_convergence(max_iterations = max_iterations )
        return self

    def widen_sample(self, max_tries = 6):
        'If the estimated min falls out of the range, adjust the range. '
        pass

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

def v_func(x, lft =10, rht =4):
    min_pt = 1
    kcenter = 0
    if abs(x - kcenter) < 1e-4:
        return min_pt
    if x < kcenter:
        return -lft * x + 1
    if x > kcenter:
        return rht * x + 1

# A min outside the search range
def almost_lin( x, b = -10, a = 3.01, noise =3.0):
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

    opt = OneDimOpt(range_min = -1, range_max= 1, initial_guess = init_start)
    opt.narrow_sample_to_converge(initial_sample= 11,  max_iterations = 40, target_function = v_func)
    # if opt.converge_flag:

    sg = plot_search_grid(opt.search_grid, opt.eval_fit(opt.quadratic_coeff), opt.colors_grid) #bokeh.palettes.Viridis11) # opt.colors_grid)
    rs = plot_residuals(opt.residuals)  
    show(column(sg, rs))

    sys.exit(0)

    def test1():
        opt = OneDimOpt(range_min = -8, range_max=8)
        # Create some widely-spaced starting points, to broaden search over possible local optima. 
        opt.init_grid()
        opt.init_pts(40)
        # Note that accuracy will improve if points distant from the estimated optimum are pruned 
        #as more nearby points are added. 
        if OneDimOpt.DBG_LVL > 0:
            print(opt.search_grid)
        est_pts = opt.run_to_convergence()


    def test2():
        # This is how to create more points
        for pt in np.linspace(0.25+ opt.range_min, opt.range_max, 1):
            opt.add_f_to_grid(pt)
        # Return the coefficients of a quadratic regression
        parabolic_fit = opt.fit_parabola_to_sample()
        print('Regression coefficients: {}'.format(parabolic_fit))
        # Evaluate the fit at the sample points
        print('>> Parabolic ', end='')
        est_quadratic_pts = opt.eval_fit( parabolic_fit)

        # Also run a linear regression, and compare errors. 
        linear_fit = opt.fit_line_to_sample()
        print('Regression coefficients: {}'.format(linear_fit))
        # Evaluate the fit at the sample points
        print('>> Linear ', end='')
        est_linear_pts = opt.eval_fit(linear_fit)

        # OK also try a constant regression
        coeff_const = opt.fit_constant()
        print('Regression coefficients: {}'.format(coeff_const))
        print('>> Constant ', end='')
        # Evaluate the fit at the sample points
        est_const_pts = opt.eval_fit(coeff_const)

        # Show both the search points and the best  fit
        plot_search_grid(opt.search_grid, est_linear_pts)

        # Show both the search points and the best parabolic fit
        plot_search_grid(opt.search_grid, est_quadratic_pts)
