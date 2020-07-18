# Introducing basic concepts of Linear Algebra

[See this set of videos on the Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

Our starting point is the properties of a linear transform in a vector space, to show that matrix multiplication just derives from these properties. 

## Intro - vectors and linearity

From high school algebra we know basic properties of addition and multiplication.

    a + 0 = a
    1a = a
    a + b = b + a
    ab = ba
    (a + b) + c = a + (b + c)
    a(b + c) = ab + bc

Similar axioms apply for vectors, with one exception, we don't have a multiplication rule for vectors. 
Once we know how to add vectors, and scale them, then we need to go the next step. 

Define a _linear_ transform, e.g. a function F on vectors st

    F(cx + dy) == cF(x) + dF(y)

In matrix notation we'll call this _multiplication_ (because, it will turn out we derive matrix multiplication from it).

Once we have this in place (it will take a while) the entirety of linear algebra will follow. 

## Determinants

A starting point is the basic properties of computing the _signed_ volume of a paralleogram (a paralleopiped in N dim space), and show it leads to 
properties that are identical to the row transforms in Gaussian Elimination. 

- Two identical rows => |A| = 0
- scaling any row by c changes the volume by c -  a linearity condition (what about sums?)
- Add a normalization - |A| of a unit cube is 1.  

Then the determinant of a _linear transform_ is just the change in volume due to the transform.  = 