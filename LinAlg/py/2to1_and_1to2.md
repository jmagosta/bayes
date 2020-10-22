# More list basics

Here are two exercises of programs that accept lists as arguments and return lists. One takes a single list and returns two, the other does the opposite; it takes two lists and returns one.  The idea is to introduce more complicated patterns, of functions that take multiple structures as lists, and return multiples. 

## Splitting a list

The problem is to go through the items in a list of words, and split them by first letter in a..j, or k..z. Imagine this is the first step in sorting words alphabetically. 
This pattern is similar the basic "list in -- list out" pattern, except it starts by initializing two empty lists, like this:

    def split_a_list( the_list):
        a_thru_j = []
        k_thru_z = []
        for a_word in the_list:
            [...]

You need to fill in the rest. To make this more interesting, you can write a program to read in a file of text and use that as input.  As you read in each line, you'll need to split the line into individual words.  For that here's a function that splits a string of characters on whitespace characters, returning substrings of non-whitespace characters:  

    'a string of words'.split()

(See what happens if instead of passsing an empty argument you pass `split()` a substring.)

## Merging two lists

A similar problem is to start with two lists and return one. The function should create a new list by alternating between the two lists first taking a word from one, then from the other. It looks like this:

    merge_lists(['Whoever', 'happy', 'make', 'happy'], ['is', 'will','others'])

returns a list we've seen before: `[ 'Whoever', 'is', 'happy', 'will', 'make', 'others', 'happy']`.  Note that the lists may be different lengths.  

A more chalenging problem it to merge the two lists so that all words starting with "a" through "j" come first. Again, this is the first step to merging two lists in sorted order.  Interesting question: Can one write the merge function that is the inverse function to `split_a_list()`? 
