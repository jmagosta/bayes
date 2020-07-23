# Intro to composite data structures.

'''
A `list` is a new type of data, made up of a list of individual entities, each of an `atomic` type.
A list keeps its items in order. It can have any number of items, and the items can be changed and added to. 

The `[]` symbol creates a list.  It is also used to select items from the list. These are two
different uses for the same symbol.

The selection operator takes a non-negative integer that `indexes` a list can be used on the righthand or lefthand side of an assignment. 

The '+' operator also has a version that combines two lists and returns a new list. 

To add a single item to the end of a list use the `append` operator. 
'''

#  Remember the `range` operator as used in a for loop?  It's just a generator of a sequence of integers

range_list = list(range(9))

for n in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    print(n)

# But a for loop can use any list to iterate over:

af1 = ['Whoever', 'is', 'happy']
af2 = [ 'will', 'make', 'others', 'happy']

af = af1 + af2
af
af[1]
af[-3]

# Slices
af[1:]

af[:1]

# iterate through a list
for k in range(len(af)):
    print(' ', end='')
    print(af[k], end='')
print('.')

# a better way
for a_word in af:
    print(' ', end='')
    print(a_word, end='')
print('.')

# How to create the function len()?
# clue:  'bool([])'  returns False
def length(a_list):
    count_so_far = 0
    while ?:
        a_list = ?
    ?
    return count_so_far

# Let's reorder the digits up to 8, by multiplying it by 5 which is relatively prime
# and use this to re-order the words in this sentence. 

r = []
for i in range(8):
    step_size = 5             # Try 105
    x  =  (i*step_size) % 8    
    r = r + [x]  
    # print(x, end= ', ')
print(f'\n{r}')


for word_index in r:
    print(' ', end='')
    print(af[word_index], end = '')
print('!')

#  What happens to the word order if you change the step_size? 

# Try this:
# Is there a choice of step_size that un-scrambles the sentence? 
# How would you write a program to find out? 

# Computing phenotypes from genotypes and vice versa
#     