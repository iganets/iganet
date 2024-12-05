# Tutorial 02: Working with B-spline functions

## TL;DR

In this tutorial you will learn how to create and manipulate B-spline
function objects and to evaluate them and their derivatives.

## Creating univariate B-spline function objects

Let us start by creating a *univariate uniform B-spline function* of
degree \f$ p=2 \f$ with 6 control points in \f$ \mathbb{R}^1 \f$

\snippet tutorial02.cxx Univariate uniform B-spline of degree 2

That is, a function \f$ C : [0,1] \to \mathbb{R}^1 \f$ that is defined
through its 6 control points \f$ x_k \in \mathbb{R}^1, k=0, \dots, 5
\f$, i.e.

\f[
  C(\xi) = \sum_{i=0}^5 = x_k B_{i,2}(\xi)
\f]

Here, univariate refers to the single variable \f$ \xi \f$ and uniform
means that the knot vector \f$ \Xi = [0, 0, 0, 0.25, 0.5, 0.75, 1, 1,
1] \f$ is uniform. By default, the control points are initialized to
the so-called *Greville abscissae*
\f[
  x_k^* = \frac{1}{p-1}\left(\xi_{i+1} + \cdots + \xi_{i+p-1} \right), \quad k=0, \dots, 5
\f]

All IgANets object have the member function `pretty_print(std::ostream
&os)` and an overload of the `operator<<` so that the curve \f$ C \f$
can be printed as follows

\snippet tutorial02.cxx Print univariate uniform B-spline of degree 2

yielding the output
```
[INFO] iganet::BSplineCommon<iganet::UniformBSplineCore<double, (short)1, (short)2>>(
parDim = 1, geoDim = 1, degrees = 2, knots = 9, coeffs = 6, options = TensorOptions(dtype=double, device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))
)
```

Here

- `parDim = 1` is for the dimension of the parameter domain \f$
[0,1]^1 \f$,

- `geoDim = 1` is the dimension of the geometric domain \f$ C \subset
\mathbb{R}^1 \f$,

- `degrees = 2` is the degree of the B-spline function,

- `knots = 9` is the number of knots in the knot vector \f$ \Xi \f$, and

- `coeffs = 6` is the number of coefficients \f$ c_k \f$.

The general relation for an open knot vector is `knots = coeffs +
degrees + 1`.

A more verbose output that also prints the knot and coefficient
vectors can be produced by

\snippet tutorial02.cxx Print verbose univariate uniform B-spline of degree 2

yielding verbose output

```
[INFO] iganet::BSplineCommon<iganet::UniformBSplineCore<double, (short)1, (short)2>>(
parDim = 1, geoDim = 1, degrees = 2, knots = 9, coeffs = 6, options = TensorOptions(dtype=double, device=cpu, layout=Strided (default), requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt))
knots [ owns/cont ] = std::__1::array<at::Tensor, 1ul>(
 0.0000
 0.0000
 0.0000
 0.2500
 0.5000
 0.7500
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{9} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
coeffs [ owns/cont ] = std::__1::array<at::Tensor, 1ul>(
 0.0000
 0.1250
 0.3750
 0.6250
 0.8750
 1.0000
[ CPUDoubleType{6} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
)
```

Here, `[ owns/cont ]` indicates that the underlying data is owned by
the tensor container and stored contiguously in memory.

All properties of the B-spline function object can also be addressed
individually, e.g.

\snippet tutorial02.cxx Print knots of univariate uniform B-spline of degree 2

yielding the output

```
[INFO] std::__1::array<long long, 1ul>(9)
std::__1::array<at::Tensor, 1ul>(
 0.0000
 0.0000
 0.0000
 0.2500
 0.5000
 0.7500
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{9} ]
[ TensorOptions(dtype=double, device=cpu, layout=Strided, requires_grad=false (default), pinned_memory=false (default), memory_format=(nullopt)) ]
)
```

Note that IgANets has overload of the `operator<<` for the standard
containers `std::array` and `std::vector`. If you want to address
individual properties restricted to a single parametric or geometric
dimension type

\snippet tutorial02.cxx Print knots of univariate uniform B-spline of degree 2 per dimension

to obtain the output

```
[INFO] 9
 0.0000
 0.0000
 0.0000
 0.2500
 0.5000
 0.7500
 1.0000
 1.0000
 1.0000
[ CPUDoubleType{9} ]
```

## Creating multivariate B-spline function objects

Like in the univariate case, a multivariate B-spline function object
is created by specifying the geometric dimension in the first template
parameter followed by a sequence of integers specifying the individual
degrees from which the parametric dimension is deduced. As an example,
the following code snipped creates a *bivariate B-spline surface* \f$
S : [0,1]^2 \to \mathbb{R}^3 \f$ with \f$ 6 \times 8 \f$ coefficients
and a *trivariate B-spline volume* \f$ V : [0,1]^3 \to \mathbb{R}^3
\f$ with \f$ 6 \times 8 \times 5 \f$, respectively

\snippet tutorial02.cxx Multivariate uniform B-splines

