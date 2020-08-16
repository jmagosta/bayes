# Data input and output to the disk store

The computer file system contains both python scripts and data files. Python files are text files identified with a `.py` suffix. The data files we consider are also text, typically identified with the suffix `.txt`.  Both files can be opened and edited in the programming editor, e.g. in VS Code. 

This section is about reading data from files on disk, and writing data out to disk by the program.  It's main use is to make data permanent, or more precisely to persist after a program finishes.  Alternatively one could write python files with data structures such as lists and import them into your program as a way to import data, but, preferrably reading from files that just contain data values is not tied to one language, and it lets the program determine what data structures to place the data in. Similarly writing out data as python statements is not advised, in part because python doesn't have a built-in way to do it, similarly to the ill-advised reasons for reading in data as python files.

Basic data file formats consist of one value to a line, for example, a sequence of words:

    Whoever
    is
    happy
    will
    make
    others
    happy.

### Input 

To read them in means to convert the data file contents to values, stored in the program as variables. The data file, say, named "quote.txt" doesn't specify what kind of variable(s) to create, or what to name them. That is up to the program, or more exactly up to the programmer. The assumptions the program makes must be consistent with the form of the data. Since most often the number of lines in the data file can vary, it makes sense to read in a file of this sort as a list of values. Hence we are back to writing functions that create lists. 

When a program starts, it is not aware of the data files available to it. The program must be connected to data files by "opening" the files by name. The built-in command for this, in our case is 

    fd = open('quote.txt', 'rt')

where the file is to be found in the current directory. The 'rt' indicates the file is to be read as text.  The value the function returns is a "file descriptor" - a special type by which the file is known to the program. 

If the file isn't found in the current directory the `open()` command returns a "FileNotFound" error. 

A for loop that iterates line by line through the file looks like

    for a_line in fd.readlines():
        ..do something to each line. 

This introduces a new syntax; a variable separated by a ".", followed by a function name. In a sense the function "belongs" to the variable. As in this case, `readlines()` knows which file to read from by its connection to the file descriptor variable. It also knows when to stop reading once it gets to the file's end. We've seen a similar dot notation with `math.log()` where the function `log()` "belongs" to the package `math`, but in this case it's a variable not a package to which the function belongs. 

    read_in_list = []
    with open('quote.txt', 'rt') as fd:
        for a_line in fd.readlines():
            read_in_list += [a_line[:-1]]


Exercise:

- Why the brackets around [a_line[:-1]]?  See what happens if you remove them. 
- Why the index [:-1] applied to a_line? See what happens if you remove it. 
- The readlines() function returns a string.  How would you write a routine that read in a list as numbers? 

The `with` statement is a convenience that creates a block in which `fd` is in scope.  Once the block is exited, then fd doesn't exist and the file is closed.  The same could be accomplished without the `with` statement by calling 

    fd.close()

once the for-loop completes, but the `with` statement also makes sure to close the file should the program fail before reaching the `close()` function. 

### Output

The inverse to the read example above uses the `write()` function that belongs to the file descriptor.

    with open('quote_out.txt', 'wt') as fd_out:
        for a_line in read_in_list:
            fd_out.write(f'{a_line}\n')

Note that we have to tell that the file is open for writing, by the argument 'wt'. Be careful that if you write to a file that already exists it will overwrite the contents of the file. If this is not what you want, open the file with the argument 'at' and the new contents will be appended to the file. 

- Exercise:

The print statement can be used to write to a file by using its optional `file=` argument.  Revise the previous example to use `print()` instead of `write()`. 

