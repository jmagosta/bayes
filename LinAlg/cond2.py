# cond2.py
# 30 Jun 2020
# More examples of using if-then-else

import math

def hinge_function(x):
    if x > 100:
        y  = x -90
    elif x > 10:
        y = 10
    elif x >= 0:
        y = x
    else:
        y = 0
    return y

def tax_rate(income, filer_type):
    if filer_type == 'single':
        if income < 9700 :
            tax = income * 0.10
        elif income < 39475:
            tax = ...

        else:
            tax = income * 0.37
    else:    # "married
        if income < 19400:
            tax = ...
        elif     income < 78950:
            pass
    return income



print(hinge_function(101))

