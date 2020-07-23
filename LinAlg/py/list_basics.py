# Basic exercises that use lists

# Lists are a way to group sequences of variables together, to treat them as a unit. 
# Here are some exercises to get familar with passing lists as values to functions that return
# values as lists.

###################
# Solving a quadratic equation - you'll need to recall the formula for this. 
# To solve the quadratic equation, you need to pass in three coefficients, a, b, c in addition to 
# the value of x at which to evaluate it. The result is a pair of values, or possibly 
# indicate that there are no real-valued solutions.  

# The function header takes the value of x and a list with the parameters
def quadratic_equation_solution(y, coefficients):
    'Solve for x, for the quadratic  y = a x^2 + b x + c'
    return [x1, x2]

# This is how it is called: 
x1, x2 = quadratic_equation_solution(1.0, [2, -1, 4])

####################
# A function that modifies a list follows this general pattern: It takes the
# list passed to it, and iterates through it, to generate the new 
# list that is returned. THe function should work on list of any size 
# (Including empty lists)

# The general pattern looks like this
def modify_a_list(the_list):
    'Treat the list items one by one to create a new list'
    # Start with an empty list
    new_list = []
    for an_item in the_list:
        # Decide what to do with the item
        # and what to add to the result
        new_list = ? 
    return new_list

# This pattern can be used for example, to
# 1. reverse the order of elements in the list
# 2. create a list with elements that are only even, odd, integer, prime, etc. 
# 3. Scale the numbers in a list by a constant. 
# 4. Divide each element by a factor and replace each element by the quotient and remainder.

# This can get complicated, for example by sorting the list, or creating a collection of
# certain subsets of the list. 
