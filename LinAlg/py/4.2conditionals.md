# Conditional execution -- the "if" statement

The `if-then-else` statement as it's sometimes called  uses logical expressions to change the flow of program execution. Here is a simple example, to solve the equation  $|x| = a$ for $x$.  It has two solutions, one positive one negative. If $x < 0$ then the answer is $-x$, which is positive, but if $x>= 0$ then the answer is just $x$. So the code to compute this is 

    if x < 0:
        a = -x
    else:
        a = x

This computes the absolute value function, so it could be wrapped in a function like this:

    def abs(x):
        if x < 0:
            a = -x
        else:
            a = x
        return a

## blocks of code

These two code snippets show how indentation is used to set off `blocks` of code. Each block is initiated by a line ending in a colon, followed by lines with common indents. 
Note that the body of the `abs()` function forms a block, as shown by the indent, and the `a = ...` statements each form a block within the function block.  A program is made up entirely of a series of nested blocks. 

Here's an example where the block in a conditional statement is 3 lines:

    if n > 0:
        n = n - 1
        is_odd = (n //2 == n/2)
        print(f'It is {is_odd} n odd')

The first statement decrements n by 1, the second checks then if that result is even, then the print statement reports "True" in the case where the original value is both integer, positive, and odd. 

Python is strict about using whitespace indents - either tabs or equivalently sequences of 4 spaces to designate blocks.  All contigious lines with at the same number of indents belong to the same block.  Any "nested" blocks within a block also count as part of the enclosing block. (Including whitepace as syntax is unique to python among computer languages, although laying out code in block structure is a universal convention.)

### Excercise

Amazingly, python does not have a "sign" function. Such a function returns -1 if it's argument is negative and 1 if positive.  Write such a function like the "abs" function.

