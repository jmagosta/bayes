# Sort exercises
# JMA 6 Oct 2020
# Sorting a sequence of items is a task that is basic to many computer science 
# algorithms. A sort method needs a comparison binary operator that evaluates 
# items pair-wise and returns True or False depending on which comes before 
# the other. 
# To just check if a list is ordered by iterating through it
# takes n (actually n-1) comparisons. In total there are possibly 
# "order -n squared", actually n(n-1)/2, comparisons between n items, 
# not all necessary to for a sort -- the number needed depends on
# the efficiency of the method. 
# 
import random

def create_random_ints(how_many, max_value=5):
    'Create a list of numbers to sort'
    some_items = []
    for _ in range(how_many):
        some_items.append(random.randint(1, max_value))
        # some_items = some_items + [random.randint(1.,5)]quit
    return some_items

numbers_to_sort = create_random_ints(20)
# For example
# [2, 2, 1, 4, 5, 4, 2, 5, 1, 3, 5, 3, 5, 3, 3, 5, 4, 1, 4, 5]

# Exercise: Creat a function that sorts this list of numbers. 

# You can start with these components. 
TN = 0 
RN = 0
# # Comparison function
def in_order(one, two):
    'Is the first item less than or equal to the second?'
    return one <= two

# Swap function
def swap(one, two):
    'Return a tuple with the order of the two items reversed'
    global TN
    TN +=1
    return two, one

# How to apply the comparision function to a list
def check_if_ordered(list_of_nos):
    global RN 
    # Asssume the list is in order, 
    #is_in_order = True

    # If the "in order" test succeeds over all pairs
    # the list is ordered, but not if there's any failure 
    for m in range(len(list_of_nos)//4):
        n = 0
        while n < len(list_of_nos)-1:
            prev_item = list_of_nos[n]
            next_item = list_of_nos[n+1]
            if not in_order(prev_item, next_item):
                list_of_nos[n], list_of_nos[n+1]  = swap(prev_item, next_item)
                # is_in_order = False
            n += 1
            RN += 1
    return list_of_nos # is_in_order
        
# Two examples 
# An ordered list
# print(check_if_ordered(list(range(10))))
# An highly likely unordered list
print(numbers_to_sort)
print(check_if_ordered(numbers_to_sort))
print(RN, TN)

####################################
### Exercise:  Sort a list
#
# Clearly sorting a list takes more computation than just checking 
# the list order by making one pass through the list. Instead
# a nested iteration can be used to make all n-squared comparisons,
# but that many are not necessary. 
# >> Write a list sorting function, and have it count the number
# of times it calls in_order().

# One presumption:   Instead of just swapping two adjacents
# when they are out of order, try moving the second toward the
# front of the list until its in the "right" position
# Will this find the minimum number of swaps necessary to order the list? 












