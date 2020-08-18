# Basic exercises that use lists

# Lists are a way to group sequences of variables together, to treat them as a unit. 
# Here are some exercises to get familar with passing lists as values to functions that return
# values as lists.

###################
# Exercise 1
# Solving a quadratic equation - you'll need to recall the formula for this. 
# To solve the quadratic equation, you need to pass in a list of three coefficients, a, b, c 
# in addition to the value of x at which to evaluate it. The result is a pair of values, 
# or possibly indicate that there are no real-valued solutions.  

# Here's the template for the function  This is a simple pattern,
# list in, list out, no need for iteration. 
# The function header takes the value of x and a list with the parameters
def quadratic_equation_solution(y, coefficients):
    'Solve for x, for the quadratic  y = a x^2 + b x + c'
    
    return [x1, x2]

# This is how it is called: 
x1, x2 = quadratic_equation_solution(1.0, [2, -1, 4])

####################
# List transformation pattern 
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

# This pattern to make a pass through a list are retrun a modified a list can be used for example, to
# Exercise 2. Scale the numbers in a list by a constant. 
# Exercise 3. Reverse the order of elements in the list
# Exercise 4. given a list of numbers, create a list with its elements that are only even, odd, integer, or prime, numbers. 
# Exercise 5. Divide each element by a factor and replace each element by its quotient and remainder.

# Here's a more complicated task: 
# Exercise 6: Remove all duplicates from a list.  
# Simple case: Take the list
af2 = [ 'Whoever', 'is', 'happy', 'will', 'make', 'others', 'happy']
# Hard case: Take the output of Exercise 5 and remove all duplicates. 
