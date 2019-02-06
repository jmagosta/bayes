    def fit_line_to_sample(self):
        'Use conventional least squares to fit to the design matrix.'
        X = self.points_design_matrix()
        X = X[:, (0,1)]
        if OneDimOpt.DBG_LVL > 1:
            print('Design matrix (X^T*X)-1:\n', np.matmul(np.transpose(X),X))
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
 