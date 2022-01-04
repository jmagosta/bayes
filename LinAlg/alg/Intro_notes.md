# Introducing basic concepts of Linear Algebra

This directory contains material for instruction in Linear Algebra, roughly following the Gilbert Strang's MIT course that can be found [here on youtube.](https://www.youtube.com/watch?v=YeyrH-Oc2p4&list=PL221E2BBF13BECF6C&index=2)

[Here's set of videos on the Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

With so much data currently, it's the computation tools you need. 

As in Gilbert Strang's course, cited above, we start with Gaussian elimination, to get a grounding in one of the useful tools, showing how a ragged set of linear equations have a general solution method. By understanding this, the concepts of dimension and linear spaces can be illustrated.  

Solving by Gaussian elimination is equivalent to decomposing a matrix by $LU$ decomposition.  Other problems, specifically solving for a matrix's eigenvalues result in other decompositions. A matrix is just a useful tool for working with different problems in many dimensions-- in fact one should speak of multiple kinds of algebras, and how they translate into matrix notation. 



## Intro - vectors and linearity

Our starting point is the properties of a linear transform in a vector space, to show that matrix multiplication just derives from these properties. 

From high school algebra we know basic properties of scalar addition and multiplication, for numbers $a$, $b$ and $c$.
$$
\begin{align}
    a + 0 =& a\\
    1a =& a\\
    a + b =& b + a\\
    (a + b) + c =& a + (b + c)\\
    a(b + c) =& ab + bc\\
\end{align}
$$

A function $F$ is   _linear_ if

$$ F(cx + dy) = cF(x) + dF(y) $$

Linear algebra is about _vector spaces_, where a vector $\vec{v} = (v_1, v_2, \cdots, v_n)$ is an ordered tuple of fixed length $n$ over a _field_ of elements (e.g. the familiar scalars).  

The axioms for a _vector space_ are the same with one exception; we don't have a multiplication rule for vectors. We can add vectors element-wise, and scale them by multiplying each element by a scalar.

### Linear forms

Much of what we study consists of _linear combinations_ of vectors 

$$ \sum^n_{i=1} \alpha_i \vec{v_i} = \alpha_1 \vec{v_1} + \alpha_2 \vec{v_2} + \cdots + \alpha_n \vec{v_n}.$$

A linear combination is another vector.  For a set of vectors, all possible linear combinations, that is for all values of $\alpha_i$s make up a _subspace_, and as we shall see, possibly a subspace equal to the entire vector space. 

Similarly we can define a linear vector transform $T$ by means of a linear combination that maps any vector into another vector in a subspace.
Just as a function maps a scalar value to another scalar, a tranform maps a vector into another vector. We will see many examples of this type in Linear Algebra.

### Multiplication

In a vector space a _matrix_ can be used for the operation of a transform $T(\vec{v})$, written as $A\vec{v}$.  In matrix notation this indicates the _multiplication_ of $\vec{v}$ by $A$. It will turn out we can derive the rules from _matrix multiplication_ from linear transforms.

Unlike with scalars matrix  multiplication is non-commutative.  We will see how this corresponds to our intuition about operations in multi-dimensional spaces. 

## Gaussian Elimination

We start with an example of a set of linear equations, called the _Barnyard_ problem that gives a concrete example of a general solution to a set of linear equations by operations that decompose a matrix into the product of two matrici, $LU$. 
Once we have this in place (it will take a while) the entirety of linear algebra will follow, as the four fundamental subspaces from the solution of linear equations are revealed. 

## Matrix inverses

$LU$ decomposition can be extended using Gauss - Jordan elimination to _invert_ a matrix, giving it's inverse, $A^{-1}$. 

`Invertable` or `non-Singular` matrici

Why does a non-invertable matrix not have an inverse? Why is $A A^{-1} = I$ not possible? 

- One can't get all columns of the $I$ matrix from combination of a singular A. 
- Alternately one can find a non-zero $x$ s.t.  $Ax = 0$  
  But  $A^{-1} A x = 0$ => $x = 0$

### Dot products

Adding an operation $<\vec{u} | \vec{v} >$ to work with distances and angles in a vector space. 


## Determinants

A starting point is the basic properties of computing the _signed_ volume of a paralleogram (a paralleopiped in N dimensional space). We show it leads to properties that can be derived from the row transforms in Gaussian Elimination. 

- Two identical rows => |A| = 0
- scaling any row by c changes the volume by c -  a linearity condition (what about sums?)
- Add a normalization - |A| of a unit cube is 1.  

Then the determinant of a _linear transform_ is just the change in volume due to the transform. 

##  Linear independence, linear basis, orthogonality, dimension


Once we understand the properties of linear combinations of vectors, we can define _dimension_.

## Eigenvalues

This is the second decomposition of a matrix. 

## Overdetermined systems, pseudo inverses, least squares

Linear models in Statistics are yet another application of matrici in high dimensional spaces. 


