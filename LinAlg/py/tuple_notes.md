


## An immutable list type: tuples

A tuple is an ordered sequence like a list, but missing some functionality. Tuples are simplier; once created they cannot be changed.  Like lists they can be indexed into, but only on the right-hand side of an assignment statement.  Unlike lists the indexed value of a tuple cannot be assigned to.  So this is ok:

    a_list[1] = a_tuple[2]

but if the positions on either side of the = sign are swapped, this doesn't work:

    a_tuple[2] = a_list[1].

There are times you will need to modify a tuple.  For this you can convert it into a list, modify it, then convert it back. 

    modifyable_list = list(a_sequence)
    modifyable_list[0] = 99
    a_modified_sequence = tuple(modifiable_list)

The same pattern works with strings, by the way. 

### Tuple constructors

A "constructor" creates an aggregate object, For lists we used "[]"; for tuples we use parentheses, 
"()". Note this is a different use than for designating an argument list for a function. 
Note that "list()" and "tuple()" also work as constructors that are used to convert a variable's contents
from one type to another. An argument list 
is not a tuple.   Analogously indexing into a sequence with "[]" is also not a constructor but an "accessor". 
Indexing into tuples (and strings for that matter) with ranges "a_sequence[1:3]" and negative values "a_sequence[-1]" 
works the same as with lists. As with any kind of sequence object, a tuple can be iterated in a _for_ statement:

    for an_item in a_tuple:
        [...]a_list

>> Write the function "extract(an_item, a_tuple)" that extracts all occurences of an_item from a_tuple.  
>> Have extract return a tuple that excludes all occurences of an_item. 

Tuples can be created on either side of the equal sign.  This is useful to return multiple values
from a function. 

    def my_func():
        (item1, item2, item3) = [1,2,3]
        return (item1, item2, item3)

    (value1, value2, value3) = my_func()

The parentheses surrounding the tuple creation are optimal, e.g. this works too:

    value1, value2, value3 = my_func()

which is an implict tuple constructor that is "ephemeral" meaning it is not assigned to a variable name.


Exercise: This code to swap two values takes three statements:

    temp_variable = value_one
    value_one = value_two
    value_two = temp_variable.

    value_two, value_one  = value_one, value_two 
    

>> Show how this can be done with just one statement using tuples.
>> How would you re-arrange more than two items with tuples?  
>> How would you write a function to sort numbers in order by reordering it with swaps? 
>> How would you sort a string? 

