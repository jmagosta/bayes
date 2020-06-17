 # Deeper dive on the python print() function.

 Computer programs need to present output to be useful. They do this in various ways, printing to the terminal is one. In python not surprisingly there is a print funtion for this. `print()` is a function unlike others in that it takes a variable number of arguments.  Arguments as strings are printed verbatum, hence the name "string literals". Arguments as variables are converted to their values and printed.  More precisely, `print` prints the string representation of the values. By convention, each argument when printed is separated by a space and the final argument is followed by a newline. This note reviews some of the useful features you can use to format terminal output with `print.` Much more detail can be found in the documentation at 
https://docs.python.org/3/tutorial/inputoutput.html

Here's an example of using `print` in a program that prints a string variable, a numeric variable and string literals.   (You can try running these 3 lines as a program.)

    my_name = "Dorothy"
    miles_walked = 99.9
    print('"If we walk', miles_walked, 'miles," says', my_name, ', "we shall sometime come to someplace."')

    "If we walk 99.9 miles," says Dorothy , "we shall sometime come to someplace."



Of the 5 arguments, the first of three string literals is '"If we walk' and between each string literal is a variable. The strings are delimited by single quotes, and the arguments are separated by commas. Don't confuse these with commas or quotes within a string. They are just part of the string and are ignored by `print` although they can make it challenging to figure out the syntax.  Just removes the delimiting quotes from the strings when it prints them. The same thing with the string variable "Dorothy"; its quotes are removed when printed. As a technical point, the numeric variable is changed to a string when printed, which in this case looks just the same as the numeric value. 

You may note that a string can be delimited by either double (as in "Dorothy") or single quotes, as in the three printed strings.  They do exactly the same thing, except the one may be embedded in a string delimited by the other.  This is how the double quotes can be printed as part of the three strings. 

## Formatted string literals. 

Python gives you a more versatile and concise way to print the same thing.  For instance, one thing to fix is to remove the space between Dorothy and the comma. 
Another way to intermingle literal and variable contents is to embed variables in `formatted strings` by enclosing the variables in a string in curly braces, like this:

    print(f'"If we walk {miles_walked} miles, says {my_name}, we shall sometime come to someplace."')

The variable name in the printed string will be replaced by the its value. Note that "f-strings" are prefaced with an `f`. Among the many things you can do will them is round
numbers with many digits, by suffixing the variable name with :.{n} where "n" is the number of significant digits to round to. Add "f" to specify instead the number
of floating point fractional digits, and "e" for exponential notation.  TO show 2 digits after the decimal, for "a_variable", the format expression is `{a_variable:.2f}`. 

You can try this:

    two = 2
    print(f'The square root {pow(2, 1/2)} rounded to {two} places is {pow(2,1/2):.{two}})')

where "pow(,) raises its first argument to the power of it's second.  Note that both the value to display and the 


## "whitespace" and other special characters

There's a third string delimiter besides single and double quotes to create multi-line strings.  Three quotes of either type create strings that span multiple lines, and so you need only one print function to print consecutive lines.  

Or you can use special "escape" sequences to print multiple lines from single line statement. "Backslash n" as it's called is one of several escape sequences for "whitespace" characters, "\t" the other that's commonly used, to insert a tab, useful for aligning columns of numbers. So the special sequence "\n" (for "newline") will start a new line when printing a single string. Similarly "\t" inserts a whitespace tab symbol. These two print statements do the same thing:

    print('Go\n\n\tF\tO\tR\n\nBroke')

    print('''Go
        F
        O
        R

    Broke''')

What if you want to just print a backslash? `print('\')` fails because python interprets `\'` as an escape sequence that doesn't exist.  Use the escape sequence `\\` instead. 

## unicode - printing characters from other alphabets. 

Python3 encodes alphabets from all languages (both living and dead) in addition to those available on the standard English keyboard. New to python3, these are called "unicode" characters, and there are a lot of them, including math symbols, accented characters and emoji--142,859 at last count. The escape sequence for unicode symbols is `\u` followed by a four-digit number. For instance here's how to print out the expression for the sum of the probability of the logical "and" of A and B: 

    print('\u2211 P(A \u2227 B)')

    ∑ P(A ∧ B)

Sadly python cannot read and evaluate unicode math symbols; they are not valid, computable expressions. 

Unicode also includes a unified set of Japanese and Chinese ideagraphs, whose unicode equivalents you can look up online: 

    print('\u65E5\u672C\u8A9E')

    日本語

Try this, but for this to work, your computer needs to have unicode fonts that include these characters. 


## Assignments:

1. Print an envelope with both sender and recipient address properly laid out, using variables for common strings. If you're ambitious include addresses with foreign characters. 

2. There's an avid field of using printed characters to make pictures.  Seach online for "ascii art" to get some ideas, then come up with one of your own. The shorter the program the better, since good programmers value laziness.  (Don't be too ambitious. We will learn ways to write more concise programs later.)
