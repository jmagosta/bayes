# Evaluation and expressions in python - a review

The notion of an _expression_ is central to the idea of _evalution_. When a computer program runs, it _interprets_ the expressions that make up the program and returns their values.  An _expression_ is a sequence of _symbols_ and _operators_ that _evaluates_ to a value. Just like algebraic expressions, there are expressions that include logical values ("logical expressions"), and even combination of string, logical and numeric values. 

A _variable_ is a symbol that's evaluated by looking up it's value in memory. You can create a practically limitless variety of variables with assignment statements. A string or number is a literal symbol that "evaluates" to itself. Quotes surround strings that are literals; strings that are not quoted are interpreted as symbols. There are also some ready-made variables; we've seen one numeric variable `math.pi` in the `math` package that holds the value of "pi."

A _function_ is a symbol--giving the function name--followed by a pair of parentheses containing it's _arguments_.  There can be zero or more arguments separated by commas. One zero-argument function is `quit()`. It ends the python session. `math.sqrt()` takes one argument. `pow(,)` raises its first argument to the power of it's second argument. 

Evaluating a composite expression happens "from the inside out", by first evaluating the individual symbols.  Any function whose arguments are symbols can then be evaluated. _Operators_ (e.g. `+`, `-`, `*` and so on) are themselves a kind of function, but they are expressed syntactically by designated symbols rather than a function-symbol followed by a pair of parenthesis surrounding a list of arguments.   Parentheses are also used to indicate which operations to do first; the inside pairs are evaluated first, then their values are passed to the expressions they are enclosed in, the same rules as apply with algebra. 

## Types 

Values are categorized into _types_.   The basic types are _integers_, _floats_ (floating point numbers), logical values (the _boolean_ values, `True` and `False`), _strings_ and _characters_. (There are many more, to be discussed later.). Each symbol has a type that is stored with it.  Types determine what are valid expressions.  Operators are type-specific.  Operators combine values of the same type, e.g. two integers add to form their sum, and two strings can be concatenated to form a longer string. Sometimes an operator can _coerce_, that is change, one type to another if there's a natural conversion. The obvious one is converting integer values to their equivalent floating point. 

So what if you _re-assign_ a variable, can it also change the type, or must the type remain unchanged?  In python both can change; this is called _dynamic typing_ unlike _strong typing_ in more formal languages.  

## Using the python prompt as a calculator

Any expression entered at the python `>>>` prompt is immediately evaluated. 
You can use this interpreter, more specifically the python "read-eval-print loop" (REPL) as a convenient printing calculator. Additionally you can define variables that can be used as calulator memory. Those variables stay in memory until your quit python.  
(See Section 3.2 on p. 13) 

What if you evaluate a variable, let's say `new_var` that hasn't been assigned? Such a variable is _unbound_, and python returns this error message:
`NameError: name 'new_var' is not defined`. Clearly the order that you enter things is important! 

When evaluating an expression, even before checking variables, python will check if what you've typed is valid. If it cannot interpret it, it returns `SyntaxError: invalid syntax`, most often just pointing to a typo you've made. 


## Exercises

- **write the solution to the quadratic expression $ax^2 + bx + c$ as two expressions---since there are two possible values it returns.** 

- Expressions can get so long that they become unreadable, but, by using variables they can be broken up into readable parts. Here's how to break up a polynomial $ax^2 + bx +c$:

    a * pow(x, 2) + b * x + c   
is equivalent to 

    (a * x + b) * x + c.   
Using an intermediate variable $v$ the same is

    v = a * x + b  
    v * x + c   
arguably not much of a saving. But it could be for higher-order polynomials.  
The infinite series for the exponential function is

    1 + x + pow(x,2)/2 + pow(x,3)/(2 * 3) + pow(x,4)/(4 * 3 * 2) + ...  
**For several terms, write this as one expression, then simplify it using intermediate variables.** 

- Here are valid expressions that demonsrate combinations of numeric, logical, and string types.  You may need to do a web search for the meaning of some of the functions, e,g, `ord()`, `chr()`, etc.  **Work out their value first manually, then check it by evaluating them at the python prompt. Then come up with a few more of your own.** 

  - Using numeric inequalities with logical operators

    (math.pi <= 4) and not (math.pi > 4)

  - Converting characters to numbers.

    (ord('A') != (ord('a'))

  - Converting numbers to characters

    'Insert'+chr(32)+'space'

  - Converting logical values to numbers

    int(False) / int(True)

  - Using format strings to convert numbers to strings

    f'X = {round(42.2)}'

  - Testing for substrings returns a logical value

    'yes' in 'yes or no'  

  - There's even an `eval()` function that takes a string and returns its value. 

    eval('math.sqrt(2.0)')

- **See also the "Try it" exercise on p.14**



